# ft_sentence_inference_iteration.py
#
# sweep_config.py의 모든 (subject, pretrain folder, fine-tuning 조합)에 대해
# ft_inference_seen_sentence.py를 실행합니다.
import os
import sys
import itertools
import subprocess

from sweep_config import SUBJECTS, SAVE_FOLDER_NAMES, FINETUNE_EXP_NAMES

skipped = []
failed = []

for sub_num, folder, ft_name in itertools.product(SUBJECTS, SAVE_FOLDER_NAMES, FINETUNE_EXP_NAMES):
    model_folder_name = os.path.join(".", "result", folder)
    ft_root = os.path.join(
        model_folder_name,
        f"leave_out_sub{sub_num}",
        ft_name,
    )
    ckpt_dir = os.path.join(
        ft_root,
        "save_model",
        f"sub{sub_num}",
        "CV0",
    )

    if not os.path.isdir(model_folder_name):
        print(f"[SKIP] pretrained 폴더 없음: {model_folder_name}")
        skipped.append((sub_num, folder, ft_name, "pretrained folder 없음"))
        continue

    if not os.path.isdir(ckpt_dir):
        print(f"[SKIP] checkpoint 폴더 없음: {ckpt_dir}")
        skipped.append((sub_num, folder, ft_name, "checkpoint folder 없음"))
        continue

    data_mat_override = f"sub{sub_num}_sentence_trial_soft_zscore.mat"

    cmd = [
        sys.executable,
        "ft_inference_seen_sentence.py",
        "--sub_num", str(sub_num),
        "--cv", "0",
        "--data_mat_override", data_mat_override,
        "--model_folder_name", model_folder_name,
        "--finetune_exp_name", ft_name,
    ]

    print("=" * 100)
    print(f"[RUN SENTENCE INF] sub{sub_num} | {folder} | {ft_name}")
    print(" ".join(cmd))
    print("=" * 100)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] sub{sub_num} | {folder} | {ft_name} | returncode={e.returncode}")
        failed.append((sub_num, folder, ft_name, e.returncode))
        continue

print("\n" + "=" * 100)
print("[ALL DONE]")
print("=" * 100)

if skipped:
    print("\n[SKIPPED LIST]")
    for x in skipped:
        print(x)

if failed:
    print("\n[FAILED LIST]")
    for x in failed:
        print(x)
