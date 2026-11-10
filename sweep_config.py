# sweep_config.py
#
# 논문에서 사용한 hyperparameter grid (pretrain LR x fine-tuning LR).
# 학습 launcher(main_iteration_LOSO_ft.py)와 inference / 결과 변환 / 요약 스크립트가 모두 이 파일을 공유합니다.
#
# 주의: 결과 재현을 위해 리스트 순서를 바꾸지 마세요.
#   - seed는 프로세스 시작 시 한 번만 설정되고, LR 조합마다 새 프로세스로 실행됩니다.
#   - pretrain 폴더마다 첫 번째 fine-tuning 조합만 pretrain과 같은 프로세스에서 실행되고,
#     이후 조합은 저장된 pretrain checkpoint를 불러와 fine-tuning만 수행합니다.

SUBJECTS = [10, 12, 13, 14, 17, 18, 21, 22, 23, 24]

BATCH_SIZE = 128
FINE_TUNE_MODE = "all"

# (max_lr, min_lr)
PRETRAIN_LR_PAIRS = [
    (3e-3, 1e-3),
    (3e-3, 2e-3),
    (3e-3, 8e-4),
    (1e-3, 1e-4),
    (1e-3, 5e-4),
    (1e-3, 8e-4),
    (8e-4, 1e-4),
    (8e-4, 3e-4),
    (8e-4, 5e-4),
    (5e-4, 1e-4),
    (5e-4, 3e-4),
]

# (max_lr, min_lr)
FINETUNE_LR_PAIRS = [
    (5e-4, 1e-4),
    (1e-3, 1e-4),
    (5e-4, 3e-4),
    (1e-3, 3e-4),
    (1e-3, 5e-4),
    (8e-4, 1e-4),
    (3e-3, 1e-4),
    (8e-4, 3e-4),
    (3e-3, 3e-4),
    (8e-4, 5e-4),
    (3e-3, 5e-4),
]


def pretrain_folder_name(min_lr, max_lr, bs=BATCH_SIZE):
    """예: P2_f10_bs128_lr_0.0001_0.0008_c4"""
    return f"P2_f10_bs{bs}_lr_{min_lr}_{max_lr}_c4"


def finetune_exp_name(min_lr, max_lr):
    """예: ft_a_0.0003_0.001 (main_I2T_leave_one_out_finetune._make_ft_exp_name 과 동일 규칙)"""
    return f"ft_a_{min_lr}_{max_lr}"


SAVE_FOLDER_NAMES = [pretrain_folder_name(min_lr, max_lr) for max_lr, min_lr in PRETRAIN_LR_PAIRS]
FINETUNE_EXP_NAMES = [finetune_exp_name(min_lr, max_lr) for max_lr, min_lr in FINETUNE_LR_PAIRS]
