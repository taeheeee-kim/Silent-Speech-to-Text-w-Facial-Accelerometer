# ft_word_inference_result_check_iteration.py
#
# fine-tuning 중 저장된 seen word test 결과(save_mat/sub{X}/CV0/test*.mat)를
# result/ft_result_eng_viseme_convert_index_with_nbest_seen.py로 viseme CSV로 변환합니다.
import os
import sys
import itertools
import subprocess

from sweep_config import SUBJECTS, SAVE_FOLDER_NAMES, FINETUNE_EXP_NAMES

skipped = []
failed = []

for sub_num, folder, ft_name in itertools.product(SUBJECTS, SAVE_FOLDER_NAMES, FINETUNE_EXP_NAMES):
    exp_root = os.path.join(
        ".",
        "result",
        folder,
        f"leave_out_sub{sub_num}",
        ft_name,
    )
    mat_dir = os.path.join(exp_root, "save_mat", f"sub{sub_num}", "CV0")

    if not os.path.isdir(mat_dir):
        print(f"[SKIP] word mat 폴더 없음: {mat_dir}")
        skipped.append((sub_num, folder, ft_name, "word mat folder 없음"))
        continue

    cmd = [
        sys.executable,
        os.path.join("result", "ft_result_eng_viseme_convert_index_with_nbest_seen.py"),
        "--sub", str(sub_num),
        "--folder_name", folder,
        "--finetune_exp_name", ft_name,
        "--cv", "0",
    ]

    print("=" * 100)
    print(f"[RUN WORD CHECK] sub{sub_num} | {folder} | {ft_name}")
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
