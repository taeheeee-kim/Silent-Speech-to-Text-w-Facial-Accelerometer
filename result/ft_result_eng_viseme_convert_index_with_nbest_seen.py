import os
import argparse
import csv
import numpy as np
import scipy.io as sio


STOI_MONO = {
    "<blank>": 0,
    "A": 1, "B": 2, "C": 3, "D": 4,
    "E": 5, "F": 6, "G": 7, "H": 8,
    "I": 9, "J": 10, "K": 11, "L": 12,
    "M": 13, "N": 14, "O": 15,
}


def extract_sequence_from_cell(cell_obj):
    if cell_obj is None:
        return None

    obj = cell_obj

    for _ in range(8):
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
            obj = obj.item()
        else:
            break

    arr = np.asarray(obj)

    if arr.size == 0:
        return None

    if arr.dtype == object:
        out = []
        for e in arr.ravel():
            if e is None:
                continue
            ee = np.asarray(e)
            if ee.size == 0:
                continue
            if ee.dtype == object:
                sub = extract_sequence_from_cell(ee)
                if sub:
                    out.extend(sub)
            else:
                out.extend([int(x) for x in ee.reshape(-1).tolist()])
        return out if len(out) > 0 else None

    arr = arr.squeeze()
    if arr.ndim == 0:
        try:
            return [int(arr)]
        except Exception:
            return [int(float(arr))]

    return [int(x) for x in arr.reshape(-1).tolist()]


def cell_array_to_sequences(cell_arr):
    if cell_arr is None or getattr(cell_arr, "size", 0) == 0:
        return []

    cell_arr = np.asarray(cell_arr)

    if cell_arr.ndim == 1:
        return [extract_sequence_from_cell(x) for x in cell_arr]

    n_r, n_c = cell_arr.shape[0], cell_arr.shape[1]
    seqs = []

    if n_r > 1 and n_c > 1:
        for i in range(n_r):
            row = cell_arr[i, :]
            seq = []
            for j in range(n_c):
                tmp = extract_sequence_from_cell(row[j])
                if tmp is not None:
                    seq.extend(tmp)
            seqs.append(seq if len(seq) > 0 else None)
    else:
        num_samples = max(n_r, n_c)
        if n_r >= n_c:
            get_cell = lambda i: cell_arr[i, 0]
        else:
            get_cell = lambda i: cell_arr[0, i]

        for i in range(num_samples):
            seqs.append(extract_sequence_from_cell(get_cell(i)))

    return seqs


def cell_array_to_nbest_sequences(cell_arr):
    if cell_arr is None or getattr(cell_arr, "size", 0) == 0:
        return []

    arr = np.asarray(cell_arr)

    if arr.ndim < 2:
        one_best = [extract_sequence_from_cell(x) for x in arr.ravel()]
        return [[s] for s in one_best]

    n_r, n_c = arr.shape[0], arr.shape[1]

    if n_r == 1 or n_c == 1:
        if n_r == 1:
            one_best = [extract_sequence_from_cell(arr[0, i, ...]) for i in range(n_c)]
        else:
            one_best = [extract_sequence_from_cell(arr[i, 0, ...]) for i in range(n_r)]
        return [[s] for s in one_best]

    nbest = []
    for i in range(n_r):
        hyps = []
        for j in range(n_c):
            seq = extract_sequence_from_cell(arr[i, j, ...])
            hyps.append(seq)
        nbest.append(hyps)

    return nbest


def cell_array_to_selected_sequences(cell_arr):
    if cell_arr is None or getattr(cell_arr, "size", 0) == 0:
        return []

    arr = np.asarray(cell_arr)
    out_all = []

    for top_cell in arr.ravel():
        if top_cell is None:
            out_all.append([])
            continue

        obj = top_cell
        for _ in range(5):
            if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
                obj = obj.item()
            else:
                break

        obj_arr = np.asarray(obj)

        if obj_arr.size == 0:
            out_all.append([])
            continue

        sample_selected = []

        if obj_arr.dtype == object:
            for subcell in obj_arr.ravel():
                seq = extract_sequence_from_cell(subcell)
                if seq is not None and len(seq) > 0:
                    sample_selected.append(seq)
        else:
            seq = extract_sequence_from_cell(obj_arr)
            if seq is not None and len(seq) > 0:
                sample_selected.append(seq)

        out_all.append(sample_selected)

    return out_all


def num_to_ipa(num_seq, stoi=STOI_MONO):
    reverse_stoi = {v: k for k, v in stoi.items()}
    return [reverse_stoi.get(int(n), "") for n in num_seq]


def ipa_to_english_pronunciation(ipa_seq):
    tokens = []
    for sym in ipa_seq:
        if sym in ["_", "<blank>"]:
            continue
        elif sym == ".":
            tokens.append(".")
        else:
            tokens.append(sym)
    return " ".join(tokens).strip()


def convert_sequence(num_seq):
    if num_seq is None:
        num_seq = []
    ipa_seq = num_to_ipa(num_seq)
    eng = ipa_to_english_pronunciation(ipa_seq)
    return ipa_seq, eng


def find_test_mat(path_load, epoch):
    candidates = [
        os.path.join(path_load, f"test{epoch}.mat"),
        os.path.join(path_load, f"test{epoch - 1}.mat"),
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError("test mat 없음:\n" + "\n".join(candidates))


def get_exp_root(base_dir, folder_name, sub, finetune_exp_name):
    if finetune_exp_name is None:
        return os.path.join(base_dir, folder_name)

    return os.path.join(
        base_dir,
        folder_name,
        f"leave_out_sub{sub}",
        finetune_exp_name,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=int, required=True)
    parser.add_argument("--folder_name", type=str, required=True)
    parser.add_argument("--finetune_exp_name", type=str, required=True)
    parser.add_argument("--cv", type=int, default=0)
    parser.add_argument("--epochs", nargs="+", type=int, default=[20,40,60,80,100,120,140,160,180,200,220,240,260,280,300])
    parser.add_argument("--base_dir", type=str, default=None,
                        help="기본값: 이 파일 위치. result 폴더 안에 두고 실행하면 보통 기본값 사용.")
    args = parser.parse_args()

    base_dir = args.base_dir or os.path.dirname(os.path.abspath(__file__))

    exp_root = get_exp_root(
        base_dir=base_dir,
        folder_name=args.folder_name,
        sub=args.sub,
        finetune_exp_name=args.finetune_exp_name,
    )

    path_load = os.path.join(
        exp_root,
        "save_mat",
        f"sub{args.sub}",
        f"CV{args.cv}",
    )

    save_path = os.path.join(path_load, "word_result_check")
    os.makedirs(save_path, exist_ok=True)

    print(f"[INFO] load dir: {path_load}")
    print(f"[INFO] save dir: {save_path}")

    for epoch in args.epochs:
        results_list = np.empty((101, 15), dtype=object)
        results_list[0, :] = [
            "pred_idx", "pred_ipa", "pred_eng",
            "best1_idx", "best1_ipa", "best1_eng",
            "best2_idx", "best2_ipa", "best2_eng",
            "best3_idx", "best3_ipa", "best3_eng",
            "target_idx", "target_ipa", "target_eng",
        ]

        try:
            mat_path = find_test_mat(path_load, epoch)
        except FileNotFoundError as e:
            print(f"[WARN] epoch {epoch}: {e}")
            continue

        data = sio.loadmat(mat_path)
        preds = data.get("preds_idx", None)
        targets = data.get("targets_idx", None)
        nbest_beam = data.get("nbest_preds_idx_beam", None)

        if preds is None or targets is None:
            print(f"[WARN] epoch {epoch}: preds_idx 또는 targets_idx 없음: {mat_path}")
            continue

        if preds.size == 0 or targets.size == 0:
            print(f"[WARN] epoch {epoch}: preds 또는 targets 비어 있음: {mat_path}")
            continue

        pred_seqs = cell_array_to_sequences(preds)
        tar_seqs = cell_array_to_sequences(targets)
        nbest_seqs = cell_array_to_nbest_sequences(nbest_beam)

        num_samples = min(len(pred_seqs), len(tar_seqs), 100)

        for i_dat in range(num_samples):
            tmp_preds = pred_seqs[i_dat] or []
            tmp_targets = tar_seqs[i_dat] or []

            pred_ipa, pred_eng = convert_sequence(tmp_preds)
            tar_ipa, tar_eng = convert_sequence(tmp_targets)

            best1 = best2 = best3 = []
            if nbest_seqs and i_dat < len(nbest_seqs):
                hyps = nbest_seqs[i_dat] or []
                if len(hyps) >= 1 and hyps[0] is not None:
                    best1 = hyps[0]
                if len(hyps) >= 2 and hyps[1] is not None:
                    best2 = hyps[1]
                if len(hyps) >= 3 and hyps[2] is not None:
                    best3 = hyps[2]

            best1_ipa, best1_eng = convert_sequence(best1)
            best2_ipa, best2_eng = convert_sequence(best2)
            best3_ipa, best3_eng = convert_sequence(best3)

            row_idx = i_dat + 1
            results_list[row_idx, :] = [
                [tmp_preds], [pred_ipa], [pred_eng],
                [best1], [best1_ipa], [best1_eng],
                [best2], [best2_ipa], [best2_eng],
                [best3], [best3_ipa], [best3_eng],
                [tmp_targets], [tar_ipa], [tar_eng],
            ]

        csv_path = os.path.join(save_path, f"result_IPA_convert_epoch{epoch}.csv")
        compare_csv_path = os.path.join(save_path, f"result_IPA_convert_epoch{epoch}_compare.csv")
        nbest_compare_csv_path = os.path.join(save_path, f"result_IPA_convert_epoch{epoch}_compare_nbest.csv")

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(results_list)

        with open(compare_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(results_list[:, [2, 14]])

        with open(nbest_compare_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(results_list[:, [5, 8, 11, 14]])

        print(f"[DONE] epoch {epoch}: {csv_path}")


if __name__ == "__main__":
    main()
