# ft_summary_metrics_from_converted_csv_3fold.py
#
# result/{folder}/leave_out_sub{X}/{ft_name}/save_mat/sub{X}/CV0/inf_seen_sentence/sentence_result_check/
#   result_IPA_convert_epoch{ep}.csv 를 읽어 sentence set(set1/2/3 및 조합)별 정확도/PER 요약표를 만듭니다.
import os
import ast
import argparse
import itertools
import numpy as np
import pandas as pd
from Levenshtein import distance as lev_distance

from sweep_config import SUBJECTS, SAVE_FOLDER_NAMES, FINETUNE_EXP_NAMES


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SUB_NUMS = SUBJECTS
DEFAULT_SAVE_FOLDERS = SAVE_FOLDER_NAMES
DEFAULT_FINETUNE_EXP_NAMES = FINETUNE_EXP_NAMES

DEFAULT_EPOCHS = [20,40,60,80,100,120,140,160,180,200,220,240,260,280,300]


# sentence eval CSV row layout:
# set1 = first 66 words, set2 = next 64 words, set3 = last 62 words
# total = 192 words
# eval_type=sentence이면 아래 7개 set 결과를 한 summary CSV 안에 모두 저장합니다.
SENTENCE_SETS_TO_REPORT = [
    "all",
    "set1",
    "set2",
    "set3",
    "set1+2",
    "set2+3",
    "set1+3",
]

SENTENCE_SET_RANGES = {
    "all": [(0, 192)],
    "set1": [(0, 66)],
    "set2": [(66, 130)],
    "set3": [(130, 192)],
    "set1+2": [(0, 130)],
    "set2+3": [(66, 192)],
    "set1+3": [(0, 66), (130, 192)],
}

SENTENCE_SET_EXPECTED_N = {
    k: sum(end - start for start, end in ranges)
    for k, ranges in SENTENCE_SET_RANGES.items()
}

METRIC_INDEX = [
    "perfect",
    "1개 차이",
    "2개 차이",
    "best1",
    "best2",
    "best1+2",
    "best3",
    "best1+2+3",
    "best1_acc(%)",
    "best1+2_acc(%)",
    "best1+2+3_acc(%)",
    "PER",
    "N",
]

COUNT_ROWS = [
    "perfect",
    "1개 차이",
    "2개 차이",
    "best1",
    "best2",
    "best1+2",
    "best3",
    "best1+2+3",
    "N",
]

PERCENT_ROWS = [
    "best1_acc(%)",
    "best1+2_acc(%)",
    "best1+2+3_acc(%)",
]

def clean_eng_series(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
         .astype(str)
         .str.strip()
         .str.strip("[]'\" ")
    )


def flatten_eval(x):
    try:
        val = ast.literal_eval(x) if isinstance(x, str) else x
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
            val = val[0]
        if not isinstance(val, list):
            val = [val]
        return val
    except Exception:
        return [x]


def to_ipa_string(x) -> str:
    lst = flatten_eval(x)
    return "".join(map(str, lst))


def mean_per_from_ipa(df: pd.DataFrame) -> float:
    pred = df["pred_ipa"].map(to_ipa_string)
    targ = df["target_ipa"].map(to_ipa_string)

    pers = []
    for p, t in zip(pred, targ):
        t = t if t is not None else ""
        p = p if p is not None else ""
        if len(t) == 0:
            pers.append(0.0)
        else:
            pers.append(lev_distance(p, t) / len(t))

    return float(np.mean(pers)) if len(pers) else 0.0


def prepare_valid_df(csv_path: str) -> pd.DataFrame:
    """CSV를 읽고, 필요한 column을 보강한 뒤 padding row를 제거합니다."""
    df = pd.read_csv(csv_path)

    needed_cols = [
        "pred_eng", "target_eng",
        "best1_eng", "best2_eng", "best3_eng",
        "pred_ipa", "target_ipa",
    ]

    for c in needed_cols:
        if c not in df.columns:
            df[c] = ""

    df["target_eng"] = df["target_eng"].fillna("").astype(str).str.strip()
    valid_mask = df["target_eng"] != ""

    n_total = len(df)
    n_valid = int(valid_mask.sum())
    n_removed = n_total - n_valid

    if n_removed > 0:
        print(f"[INFO] {os.path.basename(csv_path)}: padding row {n_removed}개 제거 ({n_total} -> {n_valid})")

    return df.loc[valid_mask].reset_index(drop=True)


def filter_by_sentence_set(df: pd.DataFrame, sentence_set: str = "all") -> pd.DataFrame:
    """
    sentence_result_check CSV는 valid target row 기준으로
    set1 66개, set2 64개, set3 62개가 순서대로 저장되어 있다고 가정하고 필터링합니다.

    - all: 전체 valid row 사용
    - set1: row 0~65
    - set2: row 66~129
    - set3: row 130~191
    - set1+2, set2+3, set1+3 조합 지원
    """
    if sentence_set is None or sentence_set == "all":
        return df.reset_index(drop=True)

    if sentence_set not in SENTENCE_SET_RANGES:
        raise ValueError(
            f"Unknown sentence_set={sentence_set}. "
            f"Choose from {list(SENTENCE_SET_RANGES.keys())}"
        )

    n_valid = len(df)
    if n_valid < 192:
        print(
            f"[WARN] valid sentence row 수가 192보다 적습니다: {n_valid}. "
            f"sentence_set={sentence_set}는 가능한 row만 선택합니다."
        )
    elif n_valid > 192:
        print(
            f"[WARN] valid sentence row 수가 192보다 많습니다: {n_valid}. "
            f"sentence_set={sentence_set}는 앞의 192개 기준 위치로 선택합니다."
        )

    parts = []
    for start, end in SENTENCE_SET_RANGES[sentence_set]:
        if start >= n_valid:
            continue
        parts.append(df.iloc[start:min(end, n_valid)])

    if len(parts) == 0:
        out = df.iloc[0:0].copy()
    else:
        out = pd.concat(parts, axis=0)

    expected_n = SENTENCE_SET_EXPECTED_N[sentence_set]
    if len(out) != expected_n:
        print(
            f"[WARN] sentence_set={sentence_set}: 선택된 row 수 {len(out)}개 "
            f"(기대값 {expected_n}개)"
        )

    return out.reset_index(drop=True)


def compute_metrics_from_df(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {
            "perfect": 0,
            "1개 차이": 0,
            "2개 차이": 0,
            "best1": 0,
            "best2": 0,
            "best1+2": 0,
            "best3": 0,
            "best1+2+3": 0,
            "best1_acc(%)": 0,
            "best1+2_acc(%)": 0,
            "best1+2+3_acc(%)": 0,
            "PER": np.nan,
            "N": 0,
        }

    df = df.copy()
    df["pred_eng"] = clean_eng_series(df["pred_eng"])
    df["target_eng"] = clean_eng_series(df["target_eng"])
    df["best1_eng"] = clean_eng_series(df["best1_eng"])
    df["best2_eng"] = clean_eng_series(df["best2_eng"])
    df["best3_eng"] = clean_eng_series(df["best3_eng"])

    perfect_mask = df["pred_eng"] == df["target_eng"]
    best1_mask = df["best1_eng"] == df["target_eng"]
    best2_mask = df["best2_eng"] == df["target_eng"]
    best3_mask = df["best3_eng"] == df["target_eng"]

    top2_mask = best1_mask | best2_mask
    top3_mask = best1_mask | best2_mask | best3_mask

    diffs = np.array(
        [lev_distance(p, t) for p, t in zip(df["pred_eng"], df["target_eng"])],
        dtype=int,
    )

    n = int(len(df))
    best1_count = int(best1_mask.sum())
    top2_count = int(top2_mask.sum())
    top3_count = int(top3_mask.sum())

    return {
        "perfect": int(perfect_mask.sum()),
        "1개 차이": int(np.sum(diffs == 1)),
        "2개 차이": int(np.sum(diffs == 2)),
        "best1": best1_count,
        "best2": int(best2_mask.sum()),
        "best1+2": top2_count,
        "best3": int(best3_mask.sum()),
        "best1+2+3": top3_count,
        "best1_acc(%)": best1_count / n * 100.0,
        "best1+2_acc(%)": top2_count / n * 100.0,
        "best1+2+3_acc(%)": top3_count / n * 100.0,
        "PER": mean_per_from_ipa(df),
        "N": n,
    }


def compute_metrics_one_csv(csv_path: str, eval_type: str = "sentence", sentence_set: str = "all") -> dict:
    df = prepare_valid_df(csv_path)

    if eval_type == "sentence":
        df = filter_by_sentence_set(df, sentence_set=sentence_set)

    return compute_metrics_from_df(df)


def get_converted_csv_dir(base_dir, folder, sub_num, finetune_exp_name, cv, eval_type):
    exp_root = os.path.join(
        base_dir,
        "result",
        folder,
        f"leave_out_sub{sub_num}",
        finetune_exp_name,
    )

    if eval_type == "word":
        return os.path.join(
            exp_root,
            "save_mat",
            f"sub{sub_num}",
            f"CV{cv}",
            "word_result_check",
        )

    if eval_type == "sentence":
        return os.path.join(
            exp_root,
            "save_mat",
            f"sub{sub_num}",
            f"CV{cv}",
            "inf_seen_sentence",
            "sentence_result_check",
        )

    raise ValueError(f"Unknown eval_type: {eval_type}")


def make_empty_table(epochs):
    return pd.DataFrame(index=METRIC_INDEX, columns=epochs, dtype=float)


def finalize_table(table: pd.DataFrame, epochs) -> pd.DataFrame:
    table["avg"] = table[epochs].mean(axis=1, skipna=True)
    table.loc[COUNT_ROWS, epochs + ["avg"]] = table.loc[COUNT_ROWS, epochs + ["avg"]].round(0)
    table.loc[PERCENT_ROWS, epochs + ["avg"]] = table.loc[PERCENT_ROWS, epochs + ["avg"]].round(2)
    table.loc["PER", epochs + ["avg"]] = table.loc["PER", epochs + ["avg"]].round(3)
    return table


def build_table_for_one_setting(base_dir, folder, sub_num, finetune_exp_name, cv, epochs, eval_type):
    """
    word: 기존과 동일하게 metric x epoch table 반환
    sentence: all/set1/set2/set3/set1+2/set2+3/set1+3 결과를 한 table에 stack하여 반환
              columns = sentence_set, metric, epoch..., avg
    """
    csv_dir = get_converted_csv_dir(
        base_dir=base_dir,
        folder=folder,
        sub_num=sub_num,
        finetune_exp_name=finetune_exp_name,
        cv=cv,
        eval_type=eval_type,
    )

    if not os.path.isdir(csv_dir):
        print(f"[WARN] CSV 폴더 없음: {csv_dir}")
        if eval_type == "sentence":
            rows = []
            for sentence_set in SENTENCE_SETS_TO_REPORT:
                t = make_empty_table(epochs)
                t["avg"] = np.nan
                tmp = t.reset_index().rename(columns={"index": "metric"})
                tmp.insert(0, "sentence_set", sentence_set)
                rows.append(tmp)
            return pd.concat(rows, ignore_index=True)

        out = make_empty_table(epochs)
        out["avg"] = np.nan
        return out

    if eval_type == "word":
        out = make_empty_table(epochs)

        for epoch in epochs:
            csv_path = os.path.join(csv_dir, f"result_IPA_convert_epoch{epoch}.csv")

            if not os.path.exists(csv_path):
                print(f"[WARN] 파일 없음: {csv_path}")
                continue

            metrics = compute_metrics_one_csv(csv_path, eval_type="word", sentence_set="all")

            for k in METRIC_INDEX:
                out.loc[k, epoch] = metrics[k]

        return finalize_table(out, epochs)

    if eval_type == "sentence":
        tables_by_set = {sentence_set: make_empty_table(epochs) for sentence_set in SENTENCE_SETS_TO_REPORT}

        for epoch in epochs:
            csv_path = os.path.join(csv_dir, f"result_IPA_convert_epoch{epoch}.csv")

            if not os.path.exists(csv_path):
                print(f"[WARN] 파일 없음: {csv_path}")
                continue

            valid_df = prepare_valid_df(csv_path)

            for sentence_set in SENTENCE_SETS_TO_REPORT:
                set_df = filter_by_sentence_set(valid_df, sentence_set=sentence_set)
                metrics = compute_metrics_from_df(set_df)

                for k in METRIC_INDEX:
                    tables_by_set[sentence_set].loc[k, epoch] = metrics[k]

        rows = []
        for sentence_set in SENTENCE_SETS_TO_REPORT:
            table = finalize_table(tables_by_set[sentence_set], epochs)
            tmp = table.reset_index().rename(columns={"index": "metric"})
            tmp.insert(0, "sentence_set", sentence_set)
            rows.append(tmp)

        return pd.concat(rows, ignore_index=True)

    raise ValueError(f"Unknown eval_type: {eval_type}")


def safe_sheet_name(name: str, used: set) -> str:
    invalid = ["\\", "/", "*", "?", ":", "[", "]"]
    for ch in invalid:
        name = name.replace(ch, "_")

    base = name[:31]
    sheet = base
    i = 1

    while sheet in used:
        suffix = f"_{i}"
        sheet = base[:31 - len(suffix)] + suffix
        i += 1

    used.add(sheet)
    return sheet


def compact_folder_name(folder: str) -> str:
    return folder.replace("pretrained_", "")


def add_metadata(table: pd.DataFrame, args, sub_num: int, folder: str, ft_name: str) -> pd.DataFrame:
    tmp = table.reset_index().rename(columns={"index": "metric"}) if "metric" not in table.columns else table.copy()
    tmp.insert(0, "eval_type", args.eval_type)
    tmp.insert(1, "sub", sub_num)
    tmp.insert(2, "folder", folder)
    tmp.insert(3, "folder_compact", compact_folder_name(folder))
    tmp.insert(4, "finetune_exp_name", ft_name)
    return tmp


def make_avg_rows(table: pd.DataFrame, args, sub_num: int, folder: str, ft_name: str) -> list:
    rows = []

    if args.eval_type == "sentence":
        for sentence_set in SENTENCE_SETS_TO_REPORT:
            t = table.loc[table["sentence_set"] == sentence_set]
            row = {
                "eval_type": args.eval_type,
                "sentence_set": sentence_set,
                "sub": sub_num,
                "folder": folder,
                "folder_compact": compact_folder_name(folder),
                "finetune_exp_name": ft_name,
            }
            for _, r in t.iterrows():
                row[f"{r['metric']}_avg"] = r["avg"]
            rows.append(row)
        return rows

    row = {
        "eval_type": args.eval_type,
        "sub": sub_num,
        "folder": folder,
        "folder_compact": compact_folder_name(folder),
        "finetune_exp_name": ft_name,
    }
    for metric in table.index:
        row[f"{metric}_avg"] = table.loc[metric, "avg"]
    rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_type",
        type=str,
        default="sentence",
        choices=["word", "sentence"],
        help="word: seen 단어 결과, sentence: 문장 내 단어 inference 결과",
    )
    parser.add_argument("--cv", type=int, default=0)
    parser.add_argument("--base_dir", type=str, default=BASE_DIR)

    parser.add_argument("--subs", nargs="+", type=int, default=DEFAULT_SUB_NUMS)
    parser.add_argument("--epochs", nargs="+", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--folders", nargs="+", type=str, default=DEFAULT_SAVE_FOLDERS)
    parser.add_argument("--finetune_exp_names", nargs="+", type=str, default=DEFAULT_FINETUNE_EXP_NAMES)

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="기본값: _ft_summary_tables/{eval_type}",
    )

    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.base_dir, "_ft_summary_tables", args.eval_type)
    os.makedirs(out_dir, exist_ok=True)

    all_tables = {}
    long_rows = []
    avg_rows = []

    for sub_num, folder, ft_name in itertools.product(args.subs, args.folders, args.finetune_exp_names):
        print("=" * 100)
        if args.eval_type == "sentence":
            print(
                f"[SUMMARY] eval={args.eval_type} | sentence_sets={SENTENCE_SETS_TO_REPORT} | "
                f"sub{sub_num} | {folder} | {ft_name}"
            )
        else:
            print(f"[SUMMARY] eval={args.eval_type} | sub{sub_num} | {folder} | {ft_name}")
        print("=" * 100)

        table = build_table_for_one_setting(
            base_dir=args.base_dir,
            folder=folder,
            sub_num=sub_num,
            finetune_exp_name=ft_name,
            cv=args.cv,
            epochs=args.epochs,
            eval_type=args.eval_type,
        )

        key = f"sub{sub_num}__{folder}__{ft_name}"
        all_tables[key] = table

        out_csv = os.path.join(out_dir, f"{args.eval_type}_sub{sub_num}__{folder}__{ft_name}_summary.csv")
        if args.eval_type == "sentence":
            table.to_csv(out_csv, index=False, encoding="utf-8-sig")
        else:
            table.to_csv(out_csv, encoding="utf-8-sig")
        print(f"[SAVED] {out_csv}")

        long_rows.append(add_metadata(table, args, sub_num, folder, ft_name))
        avg_rows.extend(make_avg_rows(table, args, sub_num, folder, ft_name))

    out_xlsx = os.path.join(out_dir, f"all_ft_{args.eval_type}_summary.xlsx")

    used_sheet_names = set()
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for key, table in all_tables.items():
            sheet = safe_sheet_name(key, used_sheet_names)
            if args.eval_type == "sentence":
                table.to_excel(writer, sheet_name=sheet, index=False)
            else:
                table.to_excel(writer, sheet_name=sheet)

    print(f"[SAVED] {out_xlsx}")

    if len(long_rows) > 0:
        df_all_long = pd.concat(long_rows, ignore_index=True)
    else:
        df_all_long = pd.DataFrame()

    out_long_csv = os.path.join(out_dir, f"all_ft_{args.eval_type}_summary_long.csv")
    df_all_long.to_csv(out_long_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {out_long_csv}")

    df_avg = pd.DataFrame(avg_rows)
    out_avg_csv = os.path.join(out_dir, f"all_ft_{args.eval_type}_summary_avg_only.csv")
    df_avg.to_csv(out_avg_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {out_avg_csv}")

    print("[DONE]")


if __name__ == "__main__":
    main()
