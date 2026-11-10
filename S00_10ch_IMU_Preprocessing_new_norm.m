clear; close all; clc

%% ============================================================
%  User settings
% =============================================================
sub_num = 24;

% 전처리 모드 선택
% "raw"                : 전처리 없음
% "trial_zscore"       : 기존 normalize와 유사한 sensor/axis-wise z-score
% "trial_soft_zscore"  : (x-mean)/sqrt(std^2 + sig_floor^2)
% "trend_residual"     : SG low trend z-score + high residual 약하게 추가
preprocess_mode = "trial_soft_zscore";
% preprocess_mode = "trend_residual";

% trial_soft_zscore 설정
sig_floor = 0.01;

% trend_residual 설정
sg_window = 10;          % Savitzky-Golay window length, 홀수
sg_polyorder = 4;        % Savitzky-Golay polynomial order
residual_beta = 0.3;     % high-frequency residual 반영 비율
center_before_savgol = true;

% 공통 설정
eps_val = 1e-8;

% 저장 파일명
% 예: sig_flip_padd_trial_soft_zscore.mat
save_filename = sprintf('sig_flip_padd_%s.mat', preprocess_mode);

%% ============================================================
%  Path settings
% =============================================================
path_cur = cd();
path_load_imu = sprintf('%s\\data\\orig\\sub%d\\sub%d_word\\IMU', path_cur, sub_num, sub_num);
% path_load_emg = sprintf('%s\\data\\orig\\sub%d\\sub%d_word\\EMG', path_cur, sub_num, sub_num);

path_save_data = sprintf('%s\\data\\feat\\sub%d', path_cur, sub_num);
path_save_trig = sprintf('%s\\trig\\sub%d', path_cur, sub_num);

%% ============================================================
%  Basic parameters
% =============================================================
fs = 100;
t = 3;

SessNum = 6;
SenNum = 100;

max_len = t * fs;
sig_mask = true(SessNum * SenNum, fs * t);
pad_idx = ones(SessNum * SenNum, 1) * (fs * t);

sig_pad = zeros(SessNum * SenNum, fs * t, 30);
imu_list = zeros(SessNum * SenNum, fs * t, 30);

%% ============================================================
%  IMU preprocessing
% =============================================================
for i_sess = 1:SessNum
    disp(i_sess)

    for i_sen = 1:SenNum
        disp(i_sen)

        cd(path_load_imu)
        cd(strcat('session_', string(i_sess)))

        load(strcat('word', string(i_sen), '_1.mat'));
        load(strcat('word', string(i_sen), '_2.mat'));

        savedata1 = save_data1 / 16384;
        savedata2 = save_data2 / 16384;

        clear save_data1
        clear save_data2

        len1 = size(savedata1, 3);
        len2 = size(savedata2, 3);
        min_len = min(len1, len2);

        savedata1 = savedata1(:, :, 1:min_len);
        savedata2 = savedata2(:, :, 1:min_len);

        savedata_s1 = savedata1(1, 1:3, :);
        savedata_s2 = savedata1(2, 1:3, :);
        savedata_s3 = savedata1(3, 1:3, :);
        savedata_s4 = savedata1(4, 1:3, :);
        savedata_s5 = savedata1(5, 1:3, :);

        savedata_s6 = savedata2(1, 1:3, :);
        savedata_s7 = savedata2(2, 1:3, :);
        savedata_s8 = savedata2(3, 1:3, :);
        savedata_s9 = savedata2(4, 1:3, :);
        savedata_s10 = savedata2(5, 1:3, :);

        tmp_imu = squeeze([savedata_s1 savedata_s2 savedata_s3 savedata_s4 savedata_s5 ...
            savedata_s6 savedata_s7 savedata_s8 savedata_s9 savedata_s10]);

        % tmp_imu: (30, time) 형태이므로 앞 300 sample 선택 후 transpose
        % 최종 tmp_imu: (300, 30)
        tmp_imu = tmp_imu(:, 1:300)';

        clear savedata_s*

        %% ====================================================
        %  New preprocessing
        % =====================================================
        tmp_imu = preprocess_imu_trial( ...
            tmp_imu, ...
            preprocess_mode, ...
            sig_floor, ...
            sg_window, ...
            sg_polyorder, ...
            residual_beta, ...
            center_before_savgol, ...
            eps_val ...
        );

        imu_list(SenNum * (i_sess - 1) + i_sen, :, :) = tmp_imu;
    end

    sig_pad = imu_list;
end

%% ============================================================
%  Save data
% =============================================================
if ~isfolder(path_save_data)
    mkdir(path_save_data)
end

cd(path_save_data)

save(save_filename, ...
    'sig_pad', ...
    'max_len', ...
    'pad_idx', ...
    'sig_mask', ...
    'preprocess_mode', ...
    'sig_floor', ...
    'sg_window', ...
    'sg_polyorder', ...
    'residual_beta', ...
    'center_before_savgol' ...
)

fprintf('\nSaved file: %s\n', fullfile(path_save_data, save_filename));

%% ============================================================
%  Save trigger
% =============================================================
trig = 1:SenNum;
trig_idx = 1:SessNum * SenNum;
trig_idx = trig_idx';

if ~isfolder(path_save_trig)
    mkdir(path_save_trig)
end

cd(path_save_trig)
save('trig.mat', "trig_idx", "trig")


%% ============================================================
%  Local functions
% =============================================================

function out = preprocess_imu_trial( ...
    x, ...
    preprocess_mode, ...
    sig_floor, ...
    sg_window, ...
    sg_polyorder, ...
    residual_beta, ...
    center_before_savgol, ...
    eps_val ...
)
    % x: (T, C), 예: (300, 30)

    x = double(x);

    switch string(preprocess_mode)

        case "raw"
            out = x;

        case "trial_zscore"
            % 기존 normalize(tmp_imu, 1)과 유사한 sensor/axis-wise z-score
            mu = mean(x, 1);
            sd = std(x, 0, 1) + eps_val;
            out = (x - mu) ./ sd;

        case "trial_soft_zscore"
            % 핵심:
            % 기존 z-score:      (x - mean) / std
            % soft z-score:      (x - mean) / sqrt(std^2 + sig_floor^2)
            %
            % std가 매우 작은 거의 움직임 없는 channel이
            % 과도하게 커지는 것을 막기 위한 방식

            mu = mean(x, 1);
            x_centered = x - mu;

            sd = std(x_centered, 0, 1);
            denom = sqrt(sd.^2 + sig_floor.^2) + eps_val;

            out = x_centered ./ denom;

        case "trend_residual"
            % 1) sensor/axis-wise 평균 제거
            if center_before_savgol
                x_centered = x - mean(x, 1);
            else
                x_centered = x;
            end

            % 2) SG filter로 low-frequency articulatory trend 추출
            sg_window_valid = make_valid_sg_window(sg_window, size(x_centered, 1), sg_polyorder);
            low = sgolayfilt(x_centered, sg_polyorder, sg_window_valid);

            % 3) high-frequency residual = raw - trend
            high = x_centered - low;

            % 4) low trend만 sensor/axis-wise z-normalize
            low_mu = mean(low, 1);
            low_centered = low - low_mu;

            low_sd = std(low_centered, 0, 1) + eps_val;
            low_norm = low_centered ./ low_sd;

            % 5) high residual은 독립 normalize하지 않고 low_sd 기준으로 약하게 추가
            high_weak = high ./ low_sd;

            out = low_norm + residual_beta * high_weak;

        otherwise
            error('Unknown preprocess_mode: %s', preprocess_mode);
    end
end


function w = make_valid_sg_window(window_length, signal_length, polyorder)
    % Savitzky-Golay window length 보정
    % - 홀수
    % - polyorder보다 큼
    % - signal length보다 길지 않음

    w = round(window_length);

    if mod(w, 2) == 0
        w = w + 1;
    end

    if mod(signal_length, 2) == 1
        max_w = signal_length;
    else
        max_w = signal_length - 1;
    end

    if w > max_w
        w = max_w;
    end

    min_w = polyorder + 2;
    if mod(min_w, 2) == 0
        min_w = min_w + 1;
    end

    if w < min_w
        w = min_w;
    end

    if w > max_w
        error( ...
            'Invalid SG setting: signal_length=%d, window_length=%d, polyorder=%d', ...
            signal_length, window_length, polyorder ...
        );
    end
end