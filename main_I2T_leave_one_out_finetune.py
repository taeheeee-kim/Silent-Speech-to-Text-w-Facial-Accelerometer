# main_I2T_leave_one_out_finetune.py
#
# Leave-one-subject-out (LOSO) pretraining + subject-wise fine-tuning.
#   Step 1. target subject를 제외한 subject 데이터로 pretrain
#           (result/{save_folder_name}/leave_out_sub{X}/pretrain_except_sub{X}_.../save_model/G_0500 이
#            이미 있으면 pretrain을 건너뜀)
#   Step 2. target subject 데이터로 fine-tuning, eval_every epoch마다 seen word test 결과(.mat) 저장
import os
import ast
import time
import random
import argparse
import copy

import numpy as np
import pandas as pd
import scipy.io as sio

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_2d_crnn import IMU_2D_to_RNN
from train import save_result, forward_model_ctc, get_ctc_output_lengths
from utils import Dataset_pho, Dataset_pho_aug, collate_fn_ctc
from utils import CosineAnnealingWarmUpRestarts


# =========================================================
# 0. Reproducibility
#    seed는 프로세스 시작 시 한 번만 설정됩니다.
#    이후 모든 학습/평가가 같은 난수 흐름을 이어서 사용하므로,
#    실행 순서(pretrain 수행 여부, subject 순서)가 바뀌면 결과도 달라집니다.
# =========================================================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================================================
# 1. Model builder
# =========================================================
def build_model(args):
    if args.model_name == "crnn":
        return IMU_2D_to_RNN(
            num_classes=args.nPho,
            n_ch=args.nCh,
            kt1=args.kt1,
            kt2=args.kt2,
            kt3=args.kt3,
            k1=args.k1,
            k2=args.k2,
            k3=args.k3,
            k4=args.k4,
            k5=args.k5,
            rnn_hidden=args.d_model,
            rnn_layers=args.rnn_layers,
            dropout=args.conv_dropout,
        )

    raise ValueError(f"Unsupported model_name: {args.model_name}")


# =========================================================
# 2. Data loading
# =========================================================
def load_pho_seq(args):
    df = pd.read_csv(
        args.pho_path,
        converters={"index_seq": ast.literal_eval}
    )
    return np.array(df["index_seq"].tolist(), dtype=object)


def load_subject_train_data(args, sub):
    mat_path = os.path.join(
        args.path_to_data_folder,
        "feat",
        f"sub{sub}",
        "sig_flip_padd_trial_soft_zscore.mat"
    )

    temp = sio.loadmat(mat_path)

    sig = temp["sig_pad"]
    src_mask = temp["sig_mask"][:, np.newaxis, :]
    pad_idx = temp["pad_idx"].astype("int32")
    max_len = int(temp["max_len"].item())

    return sig, src_mask, pad_idx, max_len


def load_subject_seen_test_data(args, sub):
    mat_path = os.path.join(
        args.path_to_data_folder,
        "feat",
        f"sub{sub}",
        "sig_flip_padd_seen_1_trial_soft_zscore.mat"
    )

    temp = sio.loadmat(mat_path)

    sig = temp["sig_pad"]
    src_mask = temp["sig_mask"][:, np.newaxis, :]
    pad_idx = temp["pad_idx"].astype("int32")
    max_len = int(temp["max_len"].item())
    te_idx = temp["s_nums"].astype("int32").squeeze() - 1

    return sig, src_mask, pad_idx, max_len, te_idx


def make_pretrain_loader(args, pho_seq, train_idx, pretrain_subjects=None):
    """
    Leave-one-subject-out pretraining loader.

    pretrain_subjects:
        현재 fine-tuning target subject를 제외한 subject list.
        None이면 args.subjects 전체를 사용합니다.
    """
    if pretrain_subjects is None:
        pretrain_subjects = list(args.subjects)
    else:
        pretrain_subjects = list(pretrain_subjects)

    if len(pretrain_subjects) == 0:
        raise ValueError("pretrain_subjects is empty. At least one subject is required for pretraining.")

    all_sig = []
    all_mask = []
    all_pad = []
    all_pho = []

    max_len_list = []

    print(f"[PRETRAIN SUBJECTS] {pretrain_subjects}")

    for sub in pretrain_subjects:
        sig, src_mask, pad_idx, max_len = load_subject_train_data(args, sub)

        all_sig.append(sig[train_idx])
        all_mask.append(src_mask[train_idx])
        all_pad.append(pad_idx[train_idx])
        all_pho.append(pho_seq[train_idx])

        max_len_list.append(max_len)

        print(f"[PRETRAIN DATA] sub{sub}: sig={sig.shape}, max_len={max_len}")

    global_max_len = max(max_len_list)

    all_sig = np.concatenate(all_sig, axis=0)
    all_mask = np.concatenate(all_mask, axis=0)
    all_pad = np.concatenate(all_pad, axis=0)
    all_pho = np.concatenate(all_pho, axis=0)

    args.nBin = global_max_len
    args.max_len = global_max_len

    dataset = Dataset_pho_aug(
        x=all_sig,
        y=all_pho,
        src_mask=all_mask,
        pad_idx=all_pad,
        nbin=args.nBin,
        aug_factor=args.aug_factor,
        fs=100
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_ctc
    )

    print(f"[PRETRAIN DATA] total x={all_sig.shape}, y={len(all_pho)}, nBin={args.nBin}")
    return loader


def make_subject_loaders(args, sub, pho_seq, train_idx):
    tr_sig, tr_mask, tr_pad, tr_max_len = load_subject_train_data(args, sub)
    te_sig, te_mask, te_pad, te_max_len, te_idx = load_subject_seen_test_data(args, sub)

    te_pho = np.array(pho_seq[te_idx], dtype=object)

    tr_sig = tr_sig[train_idx]
    tr_mask = tr_mask[train_idx]
    tr_pad = tr_pad[train_idx]
    tr_pho = np.array(pho_seq[train_idx], dtype=object)

    args.nBin = tr_max_len
    args.max_len = tr_max_len

    tr_dataset = Dataset_pho_aug(
        x=tr_sig,
        y=tr_pho,
        src_mask=tr_mask,
        pad_idx=tr_pad,
        nbin=args.nBin,
        aug_factor=args.ft_aug_factor,
        fs=100
    )

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_ctc
    )

    te_dataset = Dataset_pho(
        x=te_sig,
        y=te_pho,
        src_mask=te_mask,
        pad_idx=te_pad,
        nbin=te_max_len
    )

    te_loader = DataLoader(
        te_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_ctc
    )

    print(f"[FT DATA] sub{sub}: train={tr_sig.shape}, test={te_sig.shape}")

    return tr_loader, te_loader


# =========================================================
# 3. Train loop
# =========================================================
def train_ctc_epochs(
    model,
    loader,
    args,
    num_epochs,
    lr,
    min_lr,
    max_lr,
    save_dir,
    writer=None,
    tag="pretrain",
    save_epochs=None,
    eval_loader=None,
    eval_sub=None,
    eval_cv=0,
    eval_every=10,
    save_mat=True,
):
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(args.device)
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        first_cycle_steps=max(2, num_epochs),
        cycle_mult=1,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=min(args.warmup, max(0, num_epochs - 1)),
        gamma=1.0
    )

    if save_epochs is None:
        save_epochs = [num_epochs]

    for epo in range(num_epochs):
        start_time = time.time()
        total_loss = 0.0

        model.train()

        for batch_idx, (src, answ, src_mask, input_lengths, target_lengths) in enumerate(loader):
            src = src.to(args.device)
            src_mask = src_mask.to(args.device)
            input_lengths = input_lengths.to(args.device)
            answ = answ.to(args.device).view(-1)
            target_lengths = target_lengths.to(args.device)

            optimizer.zero_grad()

            prediction = forward_model_ctc(
                model,
                src,
                input_lengths=input_lengths,
                src_mask=src_mask
            )  # (T, B, C)

            T, B, C = prediction.shape

            output_lengths = get_ctc_output_lengths(
                model,
                input_lengths=input_lengths,
                max_T=T
            ).to(args.device)

            loss = F.ctc_loss(
                prediction,
                answ,
                input_lengths=output_lengths,
                target_lengths=target_lengths,
                blank=0,
                reduction="mean",
                zero_infinity=True
            )

            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            total_loss += loss.detach().cpu().item()

        scheduler.step(epo)

        avg_loss = total_loss / len(loader)

        if writer is not None:
            writer.add_scalar(f"{tag}/loss", avg_loss, epo + 1)
            writer.add_scalar(f"{tag}/lr", optimizer.param_groups[0]["lr"], epo + 1)

        elapsed = time.time() - start_time
        print(
            f"[{tag}] epoch {epo + 1:04d}/{num_epochs:04d} | "
            f"loss={avg_loss:.5f} | lr={optimizer.param_groups[0]['lr']:.7f} | "
            f"{elapsed:.2f}s"
        )

        if (epo + 1) in save_epochs:
            ckpt = {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": epo + 1,
                "args": vars(args)
            }
            torch.save(ckpt, os.path.join(save_dir, f"G_{epo + 1:04d}"))

        # Fine-tuning 결과 .mat 저장 (test{epoch}.mat)
        if save_mat and eval_loader is not None and ((epo + 1) % eval_every == 0):
            print(f"[SAVE MAT] {tag} epoch {epo + 1}")

            os.makedirs(args.path_to_save_mat, exist_ok=True)

            dummy_loss = {
                "G": np.zeros(num_epochs + 1),
                "L2": np.zeros(num_epochs + 1),
            }

            model.eval()

            save_result(
                model={"G": model},
                test_data=eval_loader,
                epo=epo + 1,
                args=args,
                sub=eval_sub,
                cv=eval_cv,
                stt=None,
                train_loss=dummy_loss,
                writer=writer,
                train=False
            )

            model.train()

    return model


# =========================================================
# 4. Fine-tuning mode
# =========================================================
def set_finetune_mode(model, mode="all"):
    """
    mode:
        all          : 전체 layer fine-tuning
        last_blocks  : time_down + rnn + fc만 fine-tuning
    """
    for p in model.parameters():
        p.requires_grad = True

    if mode == "all":
        return

    elif mode == "last_blocks":
        for p in model.parameters():
            p.requires_grad = False

        for name in ["time_down", "rnn", "fc"]:
            if hasattr(model, name):
                for p in getattr(model, name).parameters():
                    p.requires_grad = True

    else:
        raise ValueError(f"Unknown fine_tune_mode: {mode}")


# =========================================================
# 5. Main
# =========================================================
def _bool_arg(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def _make_ft_exp_name(args, ft_mode):
    if ft_mode == 'all':
        ft_mode_simple = 'a'
    elif ft_mode == 'last_blocks':
        ft_mode_simple = 'lb'
    if args.finetune_exp_name is None:
        return (
            f"ft_{ft_mode_simple}"
            f"_{args.finetune_min_lr}"
            f"_{args.finetune_max_lr}"
        )

    # 여러 fine-tuning mode를 한 번에 돌릴 때 결과 덮어쓰기 방지
    return f"{args.finetune_exp_name}_{ft_mode}"


def make_matlab_safe_args(args):
    safe = {}

    for k, v in vars(args).items():
        if v is None:
            safe[k] = ""          # None은 MATLAB 저장 불가 → 빈 문자열로 대체
        elif isinstance(v, (list, tuple)):
            safe[k] = np.array(v)
        elif isinstance(v, bool):
            safe[k] = int(v)
        else:
            safe[k] = v

    return safe


def main(args):
    os.makedirs(args.result_root, exist_ok=True)

    pho_seq = load_pho_seq(args)

    # 600개 단어 전체를 training에 사용
    train_idx = np.arange(0, 600)

    all_subjects = list(args.subjects)
    ft_modes = list(args.finetune_modes)
    force_skip_pretrain = _bool_arg(args.skip_pretrain)

    if len(all_subjects) < 2:
        raise ValueError("Leave-one-subject-out training requires at least two subjects in --subjects.")

    print("\n========== LEAVE-ONE-SUBJECT-OUT PRETRAIN + FINE-TUNING ==========")
    print(f"[ALL SUBJECTS] {all_subjects}")
    print(f"[FT MODES] {ft_modes}")

    run_records = []

    # -----------------------------------------------------
    # Outer loop: target subject
    #   target_sub 제외 subject들로 pretrain
    #   target_sub 본인 data로 fine-tuning
    # -----------------------------------------------------
    for target_sub in all_subjects:
        pretrain_subjects = [s for s in all_subjects if s != target_sub]
        pretrain_sub_str = "_".join(map(str, pretrain_subjects))

        print("\n" + "=" * 80)
        print(f"[TARGET SUBJECT] sub{target_sub}")
        print(f"[PRETRAIN SUBJECTS] {pretrain_subjects}")
        print("=" * 80)

        loop_root = os.path.join(
            args.result_root,
            args.save_folder_name,
            f"leave_out_sub{target_sub}"
        )

        # -------------------------------------------------
        # Step 1. Pretrain with target subject excluded
        # -------------------------------------------------
        pretrain_dir = os.path.join(
            loop_root,
            f"pretrain_except_sub{target_sub}_train_sub{pretrain_sub_str}"
        )
        pretrain_model_dir = os.path.join(pretrain_dir, "save_model")

        pretrained_ckpt_path = os.path.join(
            pretrain_model_dir,
            f"G_{args.pretrain_epoch:04d}"
        )

        external_ckpt = args.pretrained_ckpt is not None and str(args.pretrained_ckpt).strip() != ""

        if external_ckpt:
            pretrained_ckpt_path = args.pretrained_ckpt
            print(f"[USE EXTERNAL PRETRAINED CKPT] {pretrained_ckpt_path}")

        else:
            ckpt_exists = os.path.exists(pretrained_ckpt_path)

            if ckpt_exists:
                print(f"[AUTO SKIP PRETRAIN] Found checkpoint: {pretrained_ckpt_path}")

            elif force_skip_pretrain:
                raise FileNotFoundError(
                    "--skip_pretrain was set, but the expected checkpoint does not exist:\n"
                    f"{pretrained_ckpt_path}"
                )

            else:
                pre_args = copy.deepcopy(args)
                pre_args.current_target_subject = target_sub
                pre_args.current_pretrain_subjects = pretrain_subjects

                pretrain_log_dir = os.path.join(pretrain_dir, "logs")
                writer_pre = SummaryWriter(log_dir=pretrain_log_dir)

                pretrain_loader = make_pretrain_loader(
                    pre_args,
                    pho_seq,
                    train_idx,
                    pretrain_subjects=pretrain_subjects
                )
                model = build_model(pre_args)

                print("\n========== STEP 1: PRETRAIN ==========")
                print(f"[EXCLUDED SUBJECT] sub{target_sub}")
                print(f"[TRAIN SUBJECTS] {pretrain_subjects}")

                model = train_ctc_epochs(
                    model=model,
                    loader=pretrain_loader,
                    args=pre_args,
                    num_epochs=args.pretrain_epoch,
                    lr=args.pretrain_lr,
                    min_lr=args.pretrain_min_lr,
                    max_lr=args.pretrain_max_lr,
                    save_dir=pretrain_model_dir,
                    writer=writer_pre,
                    tag=f"pretrain_except_sub{target_sub}",
                    save_epochs=args.pretrain_save_epochs,
                    eval_loader=None,
                    save_mat=False,
                )

                pretrained_ckpt_path = os.path.join(
                    pretrain_model_dir,
                    f"G_{args.pretrain_epoch:04d}"
                )

                # pretrain_save_epochs에 마지막 epoch이 없더라도 최종 checkpoint 보장
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": args.pretrain_epoch,
                        "args": vars(pre_args),
                        "target_subject_excluded": target_sub,
                        "pretrain_subjects": pretrain_subjects,
                    },
                    pretrained_ckpt_path
                )

                writer_pre.close()
                print(f"\n[PRETRAIN DONE] saved at: {pretrained_ckpt_path}")

        # -------------------------------------------------
        # Step 2. Fine-tune target subject
        # -------------------------------------------------
        print("\n========== STEP 2: FINE-TUNING ==========")

        for ft_mode in ft_modes:
            print("\n" + "-" * 80)
            print(f"[FINE-TUNE] target=sub{target_sub}, mode={ft_mode}")
            print("-" * 80)

            ft_args = copy.deepcopy(args)
            ft_args.fine_tune_mode = ft_mode
            ft_args.current_target_subject = target_sub
            ft_args.current_pretrain_subjects = pretrain_subjects

            ft_exp_name = _make_ft_exp_name(args, ft_mode)

            ft_dir = os.path.join(loop_root, ft_exp_name)

            ft_args.path_to_save_folder = ft_dir
            ft_args.path_to_save_model = os.path.join(
                ft_dir, "save_model", f"sub{target_sub}", "CV0"
            ) + "/"
            ft_args.path_to_save_mat = os.path.join(
                ft_dir, "save_mat", f"sub{target_sub}", "CV0"
            ) + "/"

            os.makedirs(ft_args.path_to_save_model, exist_ok=True)
            os.makedirs(ft_args.path_to_save_mat, exist_ok=True)

            safe_args = make_matlab_safe_args(ft_args)

            sio.savemat(
                os.path.join(ft_args.path_to_save_mat, "args.mat"),
                {"args": safe_args},
                long_field_names=True
            )

            writer_ft = SummaryWriter(
                log_dir=os.path.join(ft_dir, "logs", f"sub{target_sub}")
            )

            tr_loader, te_loader = make_subject_loaders(
                ft_args,
                target_sub,
                pho_seq,
                train_idx
            )

            ft_model = build_model(ft_args)

            ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
            ft_model.load_state_dict(ckpt["model"], strict=True)

            set_finetune_mode(ft_model, mode=ft_mode)

            dummy_loss = {
                "G": np.zeros(args.finetune_epoch + 1),
                "L2": np.zeros(args.finetune_epoch + 1)
            }

            # fine-tuning 전 pretrained 상태 평가 (test0.mat)
            ft_model = ft_model.to(ft_args.device)
            ft_model.eval()

            print(f"[EVAL BEFORE FT] sub{target_sub}, mode={ft_mode}")
            save_result(
                model={"G": ft_model},
                test_data=te_loader,
                epo=0,
                args=ft_args,
                sub=target_sub,
                cv=0,
                stt=None,
                train_loss=dummy_loss,
                writer=writer_ft,
                train=False
            )

            # subject-wise fine-tuning
            ft_model = train_ctc_epochs(
                model=ft_model,
                loader=tr_loader,
                args=ft_args,
                num_epochs=args.finetune_epoch,
                lr=args.finetune_lr,
                min_lr=args.finetune_min_lr,
                max_lr=args.finetune_max_lr,
                save_dir=ft_args.path_to_save_model,
                writer=writer_ft,
                tag=f"finetune_sub{target_sub}_{ft_mode}",
                save_epochs=args.finetune_save_epochs,
                eval_loader=te_loader,
                eval_sub=target_sub,
                eval_cv=0,
                eval_every=args.eval_every,
                save_mat=True,
            )

            # fine-tuning 후 평가 (test{finetune_epoch}.mat를 다시 저장)
            # 주의: 평가 loader를 한 번 더 순회하면서 난수를 사용하므로,
            #       결과 재현을 위해 이 호출을 지우면 안 됩니다 (다음 subject 결과에 영향).
            ft_model.eval()

            print(f"[EVAL AFTER FT] sub{target_sub}, mode={ft_mode}")
            save_result(
                model={"G": ft_model},
                test_data=te_loader,
                epo=args.finetune_epoch,
                args=ft_args,
                sub=target_sub,
                cv=0,
                stt=None,
                train_loss=dummy_loss,
                writer=writer_ft,
                train=False
            )

            writer_ft.close()

            run_records.append({
                "target_sub": target_sub,
                "pretrain_subjects": " ".join(map(str, pretrain_subjects)),
                "fine_tune_mode": ft_mode,
                "pretrained_ckpt_path": pretrained_ckpt_path,
                "ft_dir": ft_dir,
                "save_model_dir": ft_args.path_to_save_model,
                "save_mat_dir": ft_args.path_to_save_mat,
            })

    summary_path = os.path.join(
        args.result_root,
        args.save_folder_name,
        "leave_one_subject_out_summary.csv"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    pd.DataFrame(run_records).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n========== ALL DONE ==========")
    print(f"[SUMMARY CSV] {summary_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--path_to_data_folder", type=str, default=os.path.join(BASE_DIR, "data"))
    parser.add_argument("--pho_path", type=str, default=os.path.join(BASE_DIR, "utils/ver3/600_word_TIMIT_mono.csv"))
    parser.add_argument("--subjects", nargs="+", type=int, default=[10, 12, 13, 14, 17, 18, 21, 22, 23, 24])

    # Save
    parser.add_argument("--result_root", type=str, default=os.path.join(BASE_DIR, "result"))
    parser.add_argument("--save_folder_name", type=str, default="P2_f10_bs128_lr_0.0001_0.0008_c4")

    # Model
    parser.add_argument("--model_name", type=str, default="crnn", choices=["crnn"])

    parser.add_argument("--nPho", type=int, default=15)
    parser.add_argument("--nCh", type=int, default=30)

    parser.add_argument("--kt1", type=int, default=21)
    parser.add_argument("--kt2", type=int, default=11)
    parser.add_argument("--kt3", type=int, default=5)

    parser.add_argument("--k1", type=int, default=64)
    parser.add_argument("--k2", type=int, default=64)
    parser.add_argument("--k3", type=int, default=128)
    parser.add_argument("--k4", type=int, default=256)
    parser.add_argument("--k5", type=int, default=256)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--conv_dropout", type=float, default=0.4)

    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    # Pretrain
    parser.add_argument("--pretrain_epoch", type=int, default=500)
    parser.add_argument("--pretrain_lr", type=float, default=1e-4)
    parser.add_argument("--pretrain_min_lr", type=float, default=1e-4)
    parser.add_argument("--pretrain_max_lr", type=float, default=8e-4)
    parser.add_argument("--pretrain_save_epochs", nargs="+", type=int, default=[100, 200, 300, 400, 500])
    parser.add_argument("--aug_factor", type=int, default=3)

    # Fine-tuning
    parser.add_argument("--finetune_epoch", type=int, default=300)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--finetune_min_lr", type=float, default=3e-4)
    parser.add_argument("--finetune_max_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_save_epochs", nargs="+", type=int, default=[20,40,60,80,100,120,140,160,180,200,220,240,260,280,300])
    parser.add_argument("--ft_aug_factor", type=int, default=3)

    # 단일 fine-tuning mode (--finetune_modes가 없을 때 사용)
    parser.add_argument("--fine_tune_mode", type=str, default="all", choices=["all", "last_blocks"])

    # 여러 fine-tuning mode를 한 번에 실행할 때 사용
    parser.add_argument("--finetune_modes", nargs="+", type=str, default=None, choices=["last_blocks", "all"])

    parser.add_argument("--eval_every", type=int, default=20)

    # save_result 옵션
    parser.add_argument("--decoder_type", type=str, default="greedy")
    parser.add_argument("--nbest", type=int, default=3)
    parser.add_argument("--beam_score_temperature", type=float, default=1.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--min_keep_hyp", type=int, default=1)
    parser.add_argument("--max_keep_hyp", type=int, default=3)

    parser.add_argument("--skip_pretrain", nargs="?", const=True, default=False)
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--finetune_exp_name", type=str, default=None)

    args = parser.parse_args()

    if args.finetune_modes is None:
        args.finetune_modes = [args.fine_tune_mode]

    main(args)
