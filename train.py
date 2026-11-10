import os
import json

import numpy as np
import scipy.io as sio
import torch

from torchaudio.models.decoder import ctc_decoder
from per import phoneme_error_rate

# ctc_decoder 용 tokens 불러오기 (실행 위치 기준 상대 경로)
try:
    with open("./utils/eng_viseme_vocab_npho15.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    itos = {int(k): v for k, v in vocab["itos"].items()}   # index → string
    blank_id = vocab["stoi"]["<blank>"]
    tokens = list(vocab["itos"].values())
except FileNotFoundError:
    print("WARNING: eng_viseme_vocab_npho15.json not found. Decoder initialization may fail.")
    tokens = ["<blank>"]
    blank_id = 0
    itos = {0: "<blank>"}

# Beam search decoder 정의
beam_decoder = ctc_decoder(
    lexicon=None,
    tokens=tokens,
    lm=None,
    nbest=3,
    beam_size=20,
    beam_size_token=5,
    blank_token="<blank>",
    sil_token="<blank>"
)


def forward_model_ctc(model, src, input_lengths=None, src_mask=None):
    """CRNN forward. 출력: (T, B, C) log-prob"""
    return model(src)


def get_ctc_output_lengths(model, input_lengths, max_T):
    """
    모델 출력 길이 계산.
    모델에 get_output_lengths가 있으면 그것을 사용.
    없으면 CRNN 구조 기준: time stride=2가 두 번 있으므로 ceil(ceil(L/2)/2).
    """
    if hasattr(model, "get_output_lengths"):
        out_lengths = model.get_output_lengths(input_lengths)
    else:
        out_lengths = (input_lengths + 3) // 4

    return out_lengths.clamp(min=1, max=max_T).long()


def ctc_greedy_decode(pred_labels, blank=0):
    """CTC greedy decoding"""
    prev = None
    result = []
    for l in pred_labels:
        if l != blank and l != prev:
            result.append(l)
        prev = l
    return result


# 결과 저장 함수
def save_result(model, test_data, epo, args, sub, cv, stt, train_loss, writer, train=True):

    print("Start Saving Result (CTC)")

    # val_per(=val_loss 계산용)은 args.decoder_type에 따라 beam/greedy 중 하나를 선택해서 채웁니다.
    # 동시에 beam/greedy 결과를 모두 저장할 수 있도록 각각의 PER도 별도로 계산합니다.
    val_per = 0.0
    val_per_beam = 0.0
    val_per_greedy = 0.0
    model['G'].eval()
    num_utts = 0

    # decoder_type에 따라 선택되어 저장되는 값들
    all_preds_idx = []      # 1-best prediction (decoder_type에 따라 beam/greedy)
    all_targets_idx = []    # target sequence
    all_nbest_idx = []      # beam n-best (greedy면 빈 리스트)

    # beam/greedy를 둘 다 저장
    all_preds_beam_idx = []     # beam 1-best
    all_nbest_beam_idx = []     # beam n-best
    all_preds_greedy_idx = []   # greedy 1-best

    with torch.no_grad():
        # collate_fn_ctc: (src, answ, src_mask, input_lengths, target_lengths)
        for src, answ, src_mask, input_lengths, target_lengths in test_data:
            src = src.to(args.device)
            src_mask = src_mask.to(args.device)
            answ = answ.to(args.device).view(-1)

            input_lengths = input_lengths.to(args.device)

            prediction = forward_model_ctc(
                model["G"],
                src,
                input_lengths=input_lengths,
                src_mask=src_mask
            )

            T = prediction.size(0)

            output_lengths = get_ctc_output_lengths(
                model["G"],
                input_lengths=input_lengths,
                max_T=T
            )

            # 현재 save_result는 batch_size=1 가정으로 작성되어 있음
            valid_T = int(output_lengths[0].item())

            prediction_valid = prediction[:valid_T]        # padding frame 제거
            log_probs = prediction_valid.transpose(0, 1).cpu()  # (B, valid_T, C)

            target_seq = answ.cpu().tolist()

            # -----------------------------
            # 1) Beam decoding (항상 수행)
            # -----------------------------
            decode_results = beam_decoder(log_probs)   # batch 차원 포함
            hyp_list = decode_results[0]               # 현재 구현은 batch=1 가정

            if len(hyp_list) == 0 or hyp_list[0].tokens.numel() == 0:
                pred_seq_beam = []
                nbest_token_idxs = []
            else:
                pred_seq_beam = hyp_list[0].tokens.cpu().tolist()
                k = getattr(args, "nbest", 3)
                k = min(k, len(hyp_list))
                nbest_token_idxs = [
                    hyp_list[i].tokens.cpu().tolist()
                    for i in range(k)
                ]

            # -----------------------------
            # 2) Greedy decoding (항상 수행)
            # -----------------------------
            pred_labels = prediction_valid.argmax(dim=-1).transpose(0, 1)  # (B, T), batch=1 가정
            seq = pred_labels[0].cpu().tolist()
            pred_seq_greedy = ctc_greedy_decode(seq, blank=blank_id)

            # -----------------------------
            # 3) PER 계산 (둘 다)
            # -----------------------------
            per_beam = phoneme_error_rate(pred_seq_beam, target_seq)
            per_greedy = phoneme_error_rate(pred_seq_greedy, target_seq)
            val_per_beam += per_beam
            val_per_greedy += per_greedy

            # -----------------------------
            # 4) decoder_type에 따른 저장 포맷
            # -----------------------------
            if args.decoder_type == 'beam':
                val_per += per_beam
                all_preds_idx.append(pred_seq_beam)
                all_nbest_idx.append(nbest_token_idxs)
            elif args.decoder_type == 'greedy':
                val_per += per_greedy
                all_preds_idx.append(pred_seq_greedy)
                all_nbest_idx.append([])
            else:
                raise ValueError(f"Unknown decoder_type: {args.decoder_type}")

            num_utts += 1
            all_targets_idx.append(target_seq)

            # -----------------------------
            # 5) 추가 저장 (beam/greedy 모두)
            # -----------------------------
            all_preds_beam_idx.append(pred_seq_beam)
            all_nbest_beam_idx.append(nbest_token_idxs)
            all_preds_greedy_idx.append(pred_seq_greedy)

    writer.add_scalar("Loss/test", val_per / len(test_data), epo)
    writer.add_scalar("Loss/test_beam", val_per_beam / len(test_data), epo)
    writer.add_scalar("Loss/test_greedy", val_per_greedy / len(test_data), epo)

    # 저장할 딕셔너리
    test = {
        'g_loss': train_loss['G'],
        'val_loss': val_per / len(test_data),
        'preds_idx': np.array(all_preds_idx, dtype=object),
        'targets_idx': np.array(all_targets_idx, dtype=object),
        'epo': epo + 1
    }

    # decoder_type에 따른 nbest
    test['nbest_preds_idx'] = np.array(all_nbest_idx, dtype=object)

    # beam/greedy 결과 모두 저장
    test['val_loss_beam'] = val_per_beam / len(test_data)
    test['val_loss_greedy'] = val_per_greedy / len(test_data)
    test['preds_idx_beam'] = np.array(all_preds_beam_idx, dtype=object)
    test['nbest_preds_idx_beam'] = np.array(all_nbest_beam_idx, dtype=object)
    test['preds_idx_greedy'] = np.array(all_preds_greedy_idx, dtype=object)

    save_path = args.path_to_save_mat + f'test{epo}.mat'
    sio.savemat(save_path, test)

    print(f"Result saved at {save_path}")
    del test
    torch.cuda.empty_cache()


def inference_for_epochs(model, args, test_data, subnum, cv, stt, writer, eval_epochs):
    """
    이미 저장된 G_XXXX 체크포인트들을 이용해서
    각 epoch에 대해 test_data에 대한 inference만 수행하고
    save_result()로 결과 .mat를 새로 저장하는 함수.
    """
    # dummy train_loss (save_result에서 g_loss를 요구하니까 형식만 맞춰줌)
    train_loss = {'G': np.zeros(args.epoch)}

    model['G'].to(args.device)

    # 저장 폴더 존재 보장
    if not os.path.isdir(args.path_to_save_mat):
        os.makedirs(args.path_to_save_mat, exist_ok=True)

    for epoch in eval_epochs:
        ckpt_path = os.path.join(args.path_to_save_model, f'G_{epoch:04d}')
        if not os.path.exists(ckpt_path):
            print(f"[WARN] checkpoint not found, skip: {ckpt_path}")
            continue

        print(f"\n=== Inference for epoch {epoch} ===")
        state = torch.load(ckpt_path, map_location=args.device)
        model['G'].load_state_dict(state['model'])
        model['G'].eval()

        # save_result의 epo는 0-based (G_0200 → epo_idx=199 → test199.mat)
        epo_idx = epoch - 1

        save_result(
            model=model,
            test_data=test_data,
            epo=epo_idx,
            args=args,
            sub=subnum,
            cv=cv,
            stt=stt,
            train_loss=train_loss,
            writer=writer,
            train=False
        )
