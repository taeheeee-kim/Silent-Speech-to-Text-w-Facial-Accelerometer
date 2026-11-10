# ft_inference_seen_sentence.py
#
# fine-tuning checkpoint(G_0020 ... G_0300)로 문장 내 단어 segment에 대해 inference하고
# {ft_dir}/save_mat/sub{X}/CV{cv}/inf_seen_sentence/test{epoch-1}.mat 로 저장합니다.
import os
import re
import ast
import random
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import AttrDict, Dataset_pho, collate_fn_ctc
from model_2d_crnn import IMU_2D_to_RNN
from train import inference_for_epochs


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _matstruct_to_dict(obj):
    if hasattr(obj, "_fieldnames"):
        return {f: _matstruct_to_dict(getattr(obj, f)) for f in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            return _matstruct_to_dict(obj.item())
        return obj
    return obj


def load_args_from_mat(args_mat_path: str) -> AttrDict:
    m = sio.loadmat(args_mat_path, struct_as_record=False, squeeze_me=True)
    if "args" not in m:
        raise KeyError(f"'args' not found in {args_mat_path}. keys={list(m.keys())}")

    d = _matstruct_to_dict(m["args"])
    clean = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.size == 1:
            v = v.item()
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        clean[k] = v

    return AttrDict(clean)


def list_all_epochs(ckpt_dir: str, allowed_epochs=None):
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    epochs = []
    for fn in os.listdir(ckpt_dir):
        m = re.match(r"^G_(\d{4})$", fn)
        if m:
            ep = int(m.group(1))
            if allowed_epochs is None or ep in allowed_epochs:
                epochs.append(ep)

    epochs = sorted(epochs)
    if len(epochs) == 0:
        raise RuntimeError(f"No checkpoints like G_???? found in: {ckpt_dir}")

    return epochs


def load_pho_seq(pho_csv_path: str, take_first_n: int = 600):
    df = pd.read_csv(pho_csv_path, converters={"index_seq": ast.literal_eval})
    pho_seq = df["index_seq"].tolist()[:take_first_n]
    return np.array(pho_seq, dtype=object)


def maybe_to_zero_based(ids: np.ndarray) -> np.ndarray:
    ids = ids.astype("int32").squeeze()
    if ids.size == 0:
        return ids
    if ids.min() >= 1:
        return ids - 1
    return ids


def parse_params_from_folder_name(save_folder_name: str, finetune_exp_name: str):
    """
    예시:
    save_folder_name = "P2_f10_bs128_lr_0.0001_0.0008_c4"
    finetune_exp_name = "ft_a_0.0001_0.001"
                    또는 "ft_lb_0.0003_0.0005"

    ft_a  -> fine_tune_mode = "all"
    ft_lb -> fine_tune_mode = "last_blocks"
    """

    base = os.path.basename(save_folder_name)

    # 1. Pretraining folder parsing
    bs_m = re.search(r"bs(\d+)", base)
    lr_m = re.search(r"lr_([0-9.eE+-]+)_([0-9.eE+-]+)", base)

    if bs_m is None or lr_m is None:
        raise ValueError(
            f"Cannot parse batch_size/pretrain lr from folder name: {save_folder_name}"
        )

    # 2. Fine-tuning folder parsing (ft_a_0.0001_0.001 / ft_lb_0.0003_0.0005)
    ft_m = re.search(
        r"^ft_(a|lb)_([0-9.eE+-]+)_([0-9.eE+-]+)$",
        finetune_exp_name
    )

    if ft_m is None:
        raise ValueError(
            f"Cannot parse fine-tuning params from finetune_exp_name: {finetune_exp_name}"
        )

    ft_mode_short = ft_m.group(1)
    finetune_min_lr = float(ft_m.group(2))
    finetune_max_lr = float(ft_m.group(3))

    if ft_mode_short == "a":
        fine_tune_mode = "all"
    else:
        fine_tune_mode = "last_blocks"

    return {
        "batch_size": int(bs_m.group(1)),
        "pretrain_min_lr": float(lr_m.group(1)),
        "pretrain_max_lr": float(lr_m.group(2)),
        "model_name": "crnn",
        "fine_tune_mode": fine_tune_mode,
        "finetune_min_lr": finetune_min_lr,
        "finetune_max_lr": finetune_max_lr,
    }


def build_fallback_args(args_cli, save_folder_name: str) -> AttrDict:
    """
    finetune 폴더에 args.mat이 없을 때 사용하는 fallback args.
    main_I2T_leave_one_out_finetune.py의 argparse default를 기준으로 구성.
    """
    parsed = parse_params_from_folder_name(save_folder_name, args_cli.finetune_exp_name)

    args = AttrDict()

    # Data
    args.path_to_data_folder = os.path.join(BASE_DIR, "data")
    args.pho_path = os.path.join(BASE_DIR, "utils", "ver3", "600_word_TIMIT_mono.csv")

    # Save
    args.result_root = os.path.join(BASE_DIR, "result")
    args.save_folder_name = save_folder_name

    # Model
    args.model_name = parsed["model_name"]
    args.nPho = 15
    args.nCh = 30
    args.nBin = 1

    args.kt1 = 21
    args.kt2 = 11
    args.kt3 = 5

    args.k1 = 64
    args.k2 = 64
    args.k3 = 128
    args.k4 = 256
    args.k5 = 256

    args.d_model = 128
    args.rnn_layers = 2
    args.conv_dropout = 0.5

    # Training
    args.batch_size = parsed["batch_size"]
    args.device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")
    args.warmup = 100
    args.weight_decay = 1e-4
    args.grad_clip = 5.0

    # Pretrain
    args.pretrain_epoch = 500
    args.pretrain_lr = 1e-4
    args.pretrain_min_lr = parsed["pretrain_min_lr"]
    args.pretrain_max_lr = parsed["pretrain_max_lr"]
    args.pretrain_save_epochs = [100, 200, 300, 400, 500]
    args.aug_factor = 3

    # Fine-tuning
    args.finetune_epoch = 300
    args.epoch = args.finetune_epoch
    args.finetune_lr = 1e-5
    args.finetune_min_lr = parsed["finetune_min_lr"]
    args.finetune_max_lr = parsed["finetune_max_lr"]
    args.finetune_save_epochs = [20,40,60,80,100,120,140,160,180,200,220,240,260,280,300]
    args.ft_aug_factor = 3
    args.fine_tune_mode = parsed["fine_tune_mode"]

    # save_result 옵션
    args.decoder_type = "beam"
    args.nbest = 3
    args.beam_score_temperature = 1.0
    args.confidence_threshold = 0.8
    args.min_keep_hyp = 1
    args.max_keep_hyp = 3

    args.skip_pretrain = True
    args.pretrained_ckpt = None
    args.finetune_exp_name = args_cli.finetune_exp_name

    return args


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

    raise ValueError(f"Unknown model_name: {args.model_name}")


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder_name", type=str, required=True,
                        help="ex) ./result/P2_f10_bs128_lr_0.0001_0.0008_c4")
    parser.add_argument("--finetune_exp_name", type=str, required=True,
                        help="ex) ft_a_0.0003_0.001")
    parser.add_argument("--sub_num", type=int, required=True)
    parser.add_argument("--cv", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_mat_override", type=str, default=None,
                        help="ex) sub10_sentence_trial_soft_zscore.mat. None이면 sig_padd_word_from_sentence.mat 사용")
    parser.add_argument("--epochs", nargs="+", type=int, default=[20,40,60,80,100,120,140,160,180,200,220,240,260,280,300])
    args_cli = parser.parse_args()

    set_seed(args_cli.seed)

    base_pretrain = args_cli.model_folder_name
    save_folder_name = os.path.basename(os.path.normpath(base_pretrain))

    base_save = os.path.join(
        base_pretrain,
        f"leave_out_sub{args_cli.sub_num}",
        args_cli.finetune_exp_name,
    )

    args_mat = os.path.join(
        base_save,
        "save_mat",
        f"sub{args_cli.sub_num}",
        f"CV{args_cli.cv}",
        "args.mat",
    )

    if os.path.isfile(args_mat):
        print(f"[INFO] load args.mat: {args_mat}")
        args = load_args_from_mat(args_mat)
        args.epoch = args.finetune_epoch
    else:
        print(f"[WARN] args.mat 없음. fallback args 사용: {args_mat}")
        args = build_fallback_args(args_cli, save_folder_name)

    args.device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")

    args.path_to_save_folder = base_save
    args.path_to_save_model = os.path.join(
        base_save, "save_model", f"sub{args_cli.sub_num}", f"CV{args_cli.cv}"
    ) + os.sep
    args.path_to_load_model = args.path_to_save_model
    args.path_to_save_mat = os.path.join(
        base_save, "save_mat", f"sub{args_cli.sub_num}", f"CV{args_cli.cv}", "inf_seen_sentence"
    ) + os.sep
    os.makedirs(args.path_to_save_mat, exist_ok=True)

    if args_cli.data_mat_override:
        mat_path = os.path.join(
            args.path_to_data_folder,
            "feat",
            f"sub{args_cli.sub_num}",
            args_cli.data_mat_override,
        )
    else:
        mat_path = os.path.join(
            args.path_to_data_folder,
            "feat",
            f"sub{args_cli.sub_num}",
            "sig_padd_word_from_sentence.mat",
        )

    print(f"[INFO] sentence data: {mat_path}")
    temp = sio.loadmat(mat_path)

    required_keys = ["sig_pad", "sig_mask", "pad_idx", "word_nums"]
    for k in required_keys:
        if k not in temp:
            raise KeyError(f"'{k}' not found in {mat_path}. keys={list(temp.keys())}")

    sig = temp["sig_pad"]
    src_mask = temp["sig_mask"][:, np.newaxis, :]
    pad_idx = temp["pad_idx"].astype("int32")
    max_len = sig.shape[1]

    word_nums = maybe_to_zero_based(temp["word_nums"])
    pho_seq = load_pho_seq(args.pho_path, take_first_n=600)
    te_pho = np.squeeze(pho_seq[word_nums])

    print("sig:", sig.shape)
    print("src_mask:", src_mask.shape)
    print("pad_idx:", pad_idx.shape)
    print("max_len / nbin:", max_len)

    te_dataset = Dataset_pho(
        x=sig[:, :, 0:args.nCh],
        y=te_pho,
        src_mask=src_mask,
        pad_idx=pad_idx,
        nbin=max_len
    )
    te_loader = DataLoader(
        te_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_ctc,
    )

    model = {"G": build_model(args)}

    writer = SummaryWriter(
        log_dir=os.path.join(
            base_save,
            "logs_inf",
            f"sub{args_cli.sub_num}",
            f"CV{args_cli.cv}",
            "seen_sentence",
        )
    )

    eval_epochs = list_all_epochs(args.path_to_save_model, allowed_epochs=set(args_cli.epochs))
    print(f"[INFO] eval epochs: {eval_epochs}")
    print(f"[INFO] ckpt dir: {args.path_to_save_model}")
    print(f"[INFO] save mat dir: {args.path_to_save_mat}")

    inference_for_epochs(
        model=model,
        args=args,
        test_data=te_loader,
        subnum=args_cli.sub_num,
        cv=args_cli.cv,
        stt=None,
        writer=writer,
        eval_epochs=eval_epochs,
    )

    writer.close()
    print(f"[DONE] saved to: {args.path_to_save_mat}")


if __name__ == "__main__":
    main()
