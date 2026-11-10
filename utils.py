import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LRScheduler


def collate_fn_ctc(batch):
    srcs, answs, src_masks, pad_idxs = zip(*batch)

    # 입력은 길이 같으니 그냥 stack
    srcs = torch.stack(srcs, dim=0)            # (B, nbin, feat_dim)
    src_masks = torch.stack(src_masks, dim=0)  # (B, 1, nbin)
    pad_idxs = torch.tensor(pad_idxs)

    # 타겟은 이어붙이기
    targets = torch.cat(answs, dim=0)
    target_lengths = torch.tensor([len(t) for t in answs], dtype=torch.long)

    # 입력 길이는 전부 nbin (=srcs.size(1)) 고정
    input_lengths = torch.full(size=(len(srcs),), fill_value=srcs.size(1), dtype=torch.long)

    return srcs, targets, src_masks, input_lengths, target_lengths


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Dataset_pho(Dataset):
    def __init__(self, x, y, src_mask, pad_idx, nbin=128):
        self.len = x.shape[0]
        self.x_data_enc = torch.FloatTensor(x)
        self.y_data = y
        self.src_mask = torch.IntTensor(src_mask)
        self.pad_idx = pad_idx
        self.nbin = nbin

    def __getitem__(self, index):
        max_start = max(0, self.pad_idx[index] - self.nbin)
        mel_start = random.randint(0, max_start)
        return (self.x_data_enc[index, mel_start:mel_start+self.nbin],
                torch.LongTensor(self.y_data[index]),
                self.src_mask[index, :, mel_start:mel_start+self.nbin],
                self.pad_idx[index])

    def __len__(self):
        return self.len


class Dataset_pho_aug(Dataset):
    def __init__(self, x, y, src_mask, pad_idx, nbin=128, aug_factor=1, fs=50, tail_sec=0.1, eps=1e-6):
        """
        aug_factor=3 설정 시:
        Data 1개당 -> [Original, TimeWarp, SpeedUp] 3개를 생성하여 학습

        변경사항
        - raw_text 제거
        - padding은 0 padding / random shift 제거
        - padding 구간은 "speed up 적용 전 원신호"의 뒷부분 tail_sec(기본 0.1초)로부터
          채널별 mean/std를 구해 Gaussian noise로 채움
        """
        self.len = x.shape[0]
        self.x_data_enc = torch.as_tensor(x, dtype=torch.float32)   # (N, T, C)
        self.y_data = y
        self.src_mask = torch.as_tensor(src_mask, dtype=torch.int32)  # (N, 1, T) 가정
        self.pad_idx = pad_idx  # 각 샘플 유효 길이
        self.nbin = nbin
        self.aug_factor = aug_factor

        self.fs = fs
        self.tail_sec = tail_sec
        self.eps = eps

    def _tail_stats(self, original_valid_data: torch.Tensor):
        """
        original_valid_data: (T, C)  - 증강 적용 전 '원신호' 유효 구간
        tail_sec(예: 0.1초) 구간에서 채널별 mean/std 반환
        """
        T, C = original_valid_data.size(0), original_valid_data.size(1)
        tail_len = max(1, int(round(self.fs * self.tail_sec)))
        tail_len = min(tail_len, T)

        tail = original_valid_data[-tail_len:, :]  # (tail_len, C)
        mean = tail.mean(dim=0)                    # (C,)
        std = tail.std(dim=0, unbiased=False)      # (C,)
        std = torch.clamp(std, min=self.eps)       # std=0 방지
        return mean, std

    def _gaussian_pad(self, data: torch.Tensor, mask: torch.Tensor, pad_needed: int, mean: torch.Tensor, std: torch.Tensor):
        """
        data: (T, C), mask: (1, T)
        pad_needed만큼 뒤에 Gaussian noise를 붙이고, mask는 0으로 패딩
        """
        if pad_needed <= 0:
            return data, mask

        C = data.size(1)
        # (pad_needed, C) 생성: randn * std + mean (채널별 브로드캐스트)
        noise = torch.randn(pad_needed, C, dtype=data.dtype) * std.unsqueeze(0) + mean.unsqueeze(0)

        out_data = torch.cat([data, noise], dim=0)  # (T+pad_needed, C)
        out_mask = F.pad(mask, (0, pad_needed), "constant", 0)  # (1, T+pad_needed)
        return out_data, out_mask

    def time_warp(self, data, mask, W=10):
        """ [Mode 1] 길이 유지, 내부 고무줄 늘리기 (안전장치 포함) """
        T = data.size(0)

        # 최소 길이 보호: center를 기준으로 좌/우가 최소 1 이상 있어야 함
        if T < 4:
            return data, mask

        # W가 너무 크면 center 선택 자체가 불가능해지므로 제한
        W = int(W)
        W = max(1, min(W, (T - 2) // 2))  # center가 1~T-2 안에 들어가도록

        # center는 양 끝(0, T) 피하기: 최소 1, 최대 T-2
        low_c = max(1, W)
        high_c = min(T - 2, T - 1 - W)
        if low_c > high_c:
            return data, mask

        center = random.randint(low_c, high_c)

        # warped도 양 끝 피하기: 최소 1, 최대 T-1
        low_w = max(1, center - W)
        high_w = min(T - 1, center + W)
        if low_w > high_w:
            return data, mask

        warped = random.randint(low_w, high_w)

        # 최종 안전장치 (혹시 모를 경우)
        warped = max(1, min(warped, T - 1))

        # Data: (T, C) -> (1, C, T)
        data_t = data.transpose(0, 1).unsqueeze(0)

        # left: 길이 warped (>=1), right: 길이 T-warped (>=1)
        left_d  = F.interpolate(data_t[:, :, :center], size=warped, mode='linear', align_corners=False)
        right_d = F.interpolate(data_t[:, :, center:], size=T - warped, mode='linear', align_corners=False)

        out_d = torch.cat([left_d, right_d], dim=2).squeeze(0).transpose(0, 1)

        # Mask: nearest (mask shape: (1, T) 가정)
        mask_u = mask.unsqueeze(0).float()  # (1, 1, T)
        left_m  = F.interpolate(mask_u[:, :, :center], size=warped, mode='nearest')
        right_m = F.interpolate(mask_u[:, :, center:], size=T - warped, mode='nearest')
        out_m = torch.cat([left_m, right_m], dim=2).squeeze(0).int()  # (1, T)

        return out_d, out_m

    def speed_up(self, data, mask, speed_range=(1.1, 1.5)):
        """ [Mode 2] 빠르게 재생(길이 축소) """
        rate = random.uniform(*speed_range)
        orig_len = data.size(0)
        new_len = int(orig_len / rate)
        if new_len < 10:
            new_len = 10  # 최소 길이 보호

        data_t = data.transpose(0, 1).unsqueeze(0)  # (1, C, T)
        out_d = F.interpolate(data_t, size=new_len, mode='linear', align_corners=False).squeeze(0).transpose(0, 1)

        mask_u = mask.unsqueeze(0).float()          # (1, 1, T)
        out_m = F.interpolate(mask_u, size=new_len, mode='nearest').squeeze(0).int()  # (1, new_len)

        return out_d, out_m

    def slight_smooth(self, data, mask, kernel_size=5):
        """
        data: (T, C)
        아주 약한 moving average smoothing
        """
        if kernel_size <= 1:
            return data, mask

        # (T, C) -> (1, C, T)
        data_t = data.transpose(0, 1).unsqueeze(0)

        # padding
        pad = kernel_size // 2
        data_pad = F.pad(data_t, (pad, pad), mode='replicate')

        weight = torch.ones(data.size(1), 1, kernel_size, dtype=data.dtype, device=data.device) / kernel_size

        # depthwise conv
        out = F.conv1d(data_pad, weight, groups=data.size(1))
        out = out.squeeze(0).transpose(0, 1)  # (T, C)

        return out, mask

    def __getitem__(self, index):
        # 1) 인덱스 -> 원본/모드
        original_idx = int(index // self.aug_factor)
        mode = int(index % self.aug_factor)  # 0:Original, 1:Warp, 2:SpeedUp

        # 2) 원본 유효 구간만 추출
        valid_len = self.pad_idx[original_idx]
        # numpy scalar/array 대응
        if hasattr(valid_len, "item"):
            valid_len = int(valid_len.item())
        else:
            valid_len = int(valid_len)
        if valid_len > self.nbin:
            valid_len = self.nbin  # 안전장치

        # 증강 적용 전 "원신호" (stats 계산용)
        orig_data = self.x_data_enc[original_idx, :valid_len, :]   # (T, C)
        orig_mask = self.src_mask[original_idx, :, :valid_len]     # (1, T)

        # tail stats는 항상 "증강 적용 전 원신호"에서 계산
        tail_mean, tail_std = self._tail_stats(orig_data)

        # 작업용 복사(참조만)
        curr_data = orig_data
        curr_mask = orig_mask

        # 3) Mode별 Augmentation 적용
        if mode == 0:
            pass

        elif mode == 1:
            w_size = max(2, int(valid_len * 0.1))
            curr_data, curr_mask = self.time_warp(curr_data, curr_mask, W=w_size)

        elif mode == 2:
            curr_data, curr_mask = self.speed_up(curr_data, curr_mask, speed_range=(1.1, 1.5))

        elif mode == 3:
            k = random.choice([3, 5])
            curr_data, curr_mask = self.slight_smooth(curr_data, curr_mask, kernel_size=k)

        # 4) 길이 nbin으로 맞추기 (패딩은 Gaussian noise, shift 없음)
        cur_len = curr_data.size(0)
        pad_needed = self.nbin - cur_len

        if pad_needed > 0:
            curr_data, curr_mask = self._gaussian_pad(
                curr_data, curr_mask, pad_needed, tail_mean, tail_std
            )
            out_valid_len = cur_len  # 패딩 제외 유효 길이

        elif pad_needed < 0:
            curr_data = curr_data[:self.nbin]
            curr_mask = curr_mask[:, :self.nbin]
            out_valid_len = self.nbin

        else:
            out_valid_len = cur_len

        return (
            curr_data,                                # (nbin, C)
            torch.as_tensor(self.y_data[original_idx], dtype=torch.long),
            curr_mask,                                # (1, nbin)
            out_valid_len
        )

    def __len__(self):
        return self.len * self.aug_factor


class CosineAnnealingWarmUpRestarts(LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.first = True
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_last_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_last_lr()):
            param_group['lr'] = lr
