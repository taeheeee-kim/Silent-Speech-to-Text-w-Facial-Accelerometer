import torch
import torch.nn as nn
import torch.nn.functional as F


class IMU_2D_to_RNN(nn.Module):
    def __init__(
        self,
        num_classes=14,
        n_ch=30,
        k1=32, k2=64, k3=128, k4=256, k5=256,
        kt1=11, kt2=7, kt3=5,
        rnn_hidden=128,
        rnn_layers=2,
        dropout=0.1,
        bidirectional=True,
        normalization="batchnorm",   # "batchnorm" or "groupnorm"
        num_groups=4,                # groupnorm일 때 사용
    ):
        super().__init__()

        def pad_same(kt):
            return (kt - 1) // 2  # 홀수 권장
        
        def get_norm_2d(c):
            if normalization.lower() == "batchnorm":
                return nn.BatchNorm2d(c)
            elif normalization.lower() == "groupnorm":
                return nn.GroupNorm(num_groups=num_groups, num_channels=c)
            else:
                raise ValueError(f"Unsupported normalization: {normalization}")

        def get_norm_1d(c):
            if normalization.lower() == "batchnorm":
                return nn.BatchNorm1d(c)
            elif normalization.lower() == "groupnorm":
                return nn.GroupNorm(num_groups=num_groups, num_channels=c)
            else:
                raise ValueError(f"Unsupported normalization: {normalization}")
        
        self.n_Ch = n_ch
        self.normalization = normalization

        # (B,1,15,300) -> (B,k1,15,300)  : H(15)는 유지, time만 컨브
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, k1, kernel_size=(1, kt1), stride=(1, 1), padding=(0, pad_same(kt1)), bias=False),
            get_norm_2d(k1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # (B,k1,15,300) -> (B,k2,15,150) : time stride=2 다운샘플
        self.conv2 = nn.Sequential(
            nn.Conv2d(k1, k2, kernel_size=(1, kt2), stride=(1, 2), padding=(0, pad_same(kt2)), bias=False),
            get_norm_2d(k2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # (B,k2,15,150) -> (B,k3,5,150)
        # H축(=15=5센서*3축)에서 3개씩 묶기: kernel_h=3, stride_h=3
        self.axis_to_sensor = nn.Sequential(
            nn.Conv2d(k2, k3, kernel_size=(3, 1), stride=(3, 1), padding=(0, 0), bias=False),
            get_norm_2d(k3),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        num_sensors = int(n_ch/3)

        # (B,k3,5,150) -> (B,k4,1,150)
        # H축(=센서 5개) 전체를 한 번에 합치기: kernel_h=5
        self.sensor_to_all = nn.Sequential(
            nn.Conv2d(k3, k4, kernel_size=(num_sensors, 1), stride=(1, 1), padding=(0, 0), bias=False),
            get_norm_2d(k4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # (B,k4,150) -> (B,k5,75) : time stride=2 (Conv1d)
        self.time_down = nn.Sequential(
            nn.Conv1d(k4, k5, kernel_size=kt3, stride=2, padding=pad_same(kt3), bias=False),
            get_norm_1d(k5),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # RNN
        self.rnn = nn.LSTM(
            input_size=k5,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        out_rnn = rnn_hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_rnn, num_classes)

    def forward(self, x, return_features=False, debug=False):
        """
        x: (B, 15, 300) 또는 (B, 300, 15) 둘 다 허용
        """
        shapes = {}

        # (B,300,15)로 들어오면 (B,15,300)으로
        if x.dim() != 3:
            raise ValueError(f"Expected (B,30,300) or (B,300,30), got {tuple(x.shape)}")
        if x.shape[1] == 300 and x.shape[2] == self.n_Ch:
            x = x.transpose(1, 2)  # (B,15,300)

        # (B,1,15,300)
        x = x.unsqueeze(1)
        if debug: shapes["(B,1,30,300)"] = tuple(x.shape)

        # (B,k1,15,300)
        x = self.conv1(x)
        if debug: shapes["(B,k1,30,300)"] = tuple(x.shape)

        # (B,k2,15,150)
        x = self.conv2(x)   
        if debug: shapes["(B,k2,30,150)"] = tuple(x.shape)

        # (B,k3,5,150)
        x = self.axis_to_sensor(x)
        if debug: shapes["(B,k3,10,150)"] = tuple(x.shape)

        # (B,k4,1,150)
        x = self.sensor_to_all(x)
        if debug: shapes["(B,k4,1,150)"] = tuple(x.shape)

        # (B,k4,150)
        x = x.squeeze(2)  # H=1 제거
        if debug: shapes["(B,k4,150)"] = tuple(x.shape)

        # (B,k5,75)
        x = self.time_down(x)
        if debug: shapes["(B,k5,75)"] = tuple(x.shape)

        # RNN 입력: (B,75,k5)
        x = x.transpose(1, 2)
        if debug: shapes["RNN_in_(B,75,k5)"] = tuple(x.shape)

        x, _ = self.rnn(x)     # (B,75,2H) or (B,75,H)
        features = x

        x = self.fc(x)         # (B,75,V)
        x = F.log_softmax(x, dim=-1)
        x = x.transpose(0, 1)  # (75,B,V) for CTC

        if debug:
            return x, features, shapes

        if return_features:
            return x, features
        return x
