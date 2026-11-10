# main_iteration_LOSO_ft.py
#
# sweep_config.py의 (pretrain LR x fine-tuning LR) 조합마다
# main_I2T_leave_one_out_finetune.py를 새 프로세스로 실행합니다.
import sys
import itertools
import subprocess

from sweep_config import BATCH_SIZE, FINE_TUNE_MODE, PRETRAIN_LR_PAIRS, FINETUNE_LR_PAIRS, pretrain_folder_name

BASE_CMD = [sys.executable, "main_I2T_leave_one_out_finetune.py"]

for (max_lr, min_lr), (finetune_max_lr, finetune_min_lr) in itertools.product(PRETRAIN_LR_PAIRS, FINETUNE_LR_PAIRS):
    save_folder_name = pretrain_folder_name(min_lr, max_lr)

    cmd = BASE_CMD + [
        f"--batch_size={BATCH_SIZE}",
        f"--pretrain_min_lr={min_lr}",
        f"--pretrain_max_lr={max_lr}",
        f"--save_folder_name={save_folder_name}",
        f"--fine_tune_mode={FINE_TUNE_MODE}",
        f"--finetune_min_lr={finetune_min_lr}",
        f"--finetune_max_lr={finetune_max_lr}",
    ]

    print("\n====================================")
    print("Running command:")
    print(" ".join(str(c) for c in cmd))
    print("====================================\n")

    result = subprocess.run(cmd)

    # 에러 나면 이후 실험 중단
    if result.returncode != 0:
        print(f"실행 중 에러 발생 (returncode={result.returncode}). 이후 실험 중단.")
        break
