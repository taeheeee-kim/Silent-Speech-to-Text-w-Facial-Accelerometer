clear; close all; clc

%% ============================================================
%  User settings
% =============================================================
sub_num = 23;

SEEN_PATH = 'session_1_seen';

% 전처리 모드 선택
% "raw"                  : 전처리 없음
% "trial_zscore"         : 기존 normalize(tmp_imu,1)과 유사
% "trial_soft_zscore"    : (x-mean)/sqrt(std^2 + sig_floor^2)
% "trend_residual"       : SG low trend z-score + high residual 약하게 추가
% "trend_residual_soft"  : SG low trend soft z-score + high residual 약하게 추가
preprocess_mode = "trial_soft_zscore";
% preprocess_mode = "trend_residual";
% preprocess_mode = "trend_residual_soft";

% soft z-score 설정
sig_floor = 0.01;

% trend_residual 설정
sg_window = 10;
sg_polyorder = 4;
residual_beta = 0.3;
center_before_savgol = true;

% 공통 설정
eps_val = 1e-8;

% 저장 파일명 설정
% 기존 이름을 유지하고 싶으면 false
% mode별로 파일을 따로 저장하고 싶으면 true
AUTO_APPEND_MODE_TO_SAVE_NAME = true;

if AUTO_APPEND_MODE_TO_SAVE_NAME
    SAVE_NAME = sprintf('sig_flip_padd_seen_1_%s.mat', preprocess_mode);
else
    SAVE_NAME = 'sig_flip_padd_seen_1.mat';
end

% 수동으로 단어 번호 설정
s_nums = 1:100;

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

SessNum = 1;
SenNum = 100;

max_len = t * fs;
sig_mask = true(SessNum * SenNum, fs * t);
pad_idx = ones(SessNum * SenNum, 1) * (fs * t);

% 기존 코드에는 15로 되어 있었지만, tmp_imu는 (300,30)이므로 30으로 수정
sig_pad = zeros(SessNum * SenNum, fs * t, 30);

%% ============================================================
%  File sorting
% =============================================================
cd(path_load_imu)
cd(SEEN_PATH)
s = dir('*.mat');

cd(path_cur)

% 파일명 예: word001_1.mat, word001_2.mat
% 1열: word 번호, 2열: sensor-device 번호 또는 pair 번호
nums_mat = zeros(length(s), 2);

for i = 1:length(s)
    tokens = regexp(s(i).name, 'word(\d+)_(\d+)\.mat', 'tokens');

    if ~isempty(tokens)
        nums_mat(i, 1) = str2double(tokens{1}{1});
        nums_mat(i, 2) = str2double(tokens{1}{2});
    else
        nums_mat(i, :) = [inf, inf];
    end
end

[~, idx] = sortrows(nums_mat, [1, 2]);
s_sorted = s(idx);

%% ============================================================
%  IMU preprocessing
% =============================================================
for i_sess = 6
    disp(i_sess)

    imu_list = zeros(SenNum, fs * t, 30);

    for i_sen = 1:SenNum
        cd(path_load_imu)
        cd(SEEN_PATH)

        disp(s_sorted(2 * i_sen - 1).name)
        load(s_sorted(2 * i_sen - 1).name);

        disp(s_sorted(2 * i_sen).name)
        load(s_sorted(2 * i_sen).name);

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

        % tmp_imu: (30, time) → 앞 300 sample 선택 후 transpose
        % 최종 tmp_imu: (300, 30)
        tmp_imu = tmp_imu(:, 1:300)';

        clear savedata_s*

        %% ====================================================
        %  기존:
        %  tmp_imu = normalize(tmp_imu,1);
        %
        %  수정:
        %  mode 선택형 전처리
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

        imu_list(i_sen, :, :) = tmp_imu;
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

save(SAVE_NAME, ...
    'sig_pad', ...
    'max_len', ...
    'pad_idx', ...
    'sig_mask', ...
    's_nums', ...
    'preprocess_mode', ...
    'sig_floor', ...
    'sg_window', ...
    'sg_polyorder', ...
    'residual_beta', ...
    'center_before_savgol' ...
);

fprintf('\nSaved file: %s\n', fullfile(path_save_data, SAVE_NAME));

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
save('trig_seen.mat', "trig_idx", "trig")


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
            % 기존 normalize(tmp_imu, 1)과 유사
            % 각 sensor/axis channel별 시간축 z-score
            mu = mean(x, 1);
            sd = std(x, 0, 1) + eps_val;
            out = (x - mu) ./ sd;

        case "trial_soft_zscore"
            % soft z-score
            % 기존 z-score:
            %   (x - mean) / std
            %
            % 수정:
            %   (x - mean) / sqrt(std^2 + sig_floor^2)
            %
            % std가 작은 거의 움직임 없는 channel이
            % 과도하게 커지는 것을 방지

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

            % 2) SG filter로 low-frequency trend 추출
            sg_window_valid = make_valid_sg_window( ...
                sg_window, ...
                size(x_centered, 1), ...
                sg_polyorder ...
            );

            low = sgolayfilt(x_centered, sg_polyorder, sg_window_valid);

            % 3) high-frequency residual
            high = x_centered - low;

            % 4) low trend만 sensor/axis-wise z-normalize
            low_mu = mean(low, 1);
            low_centered = low - low_mu;

            low_sd = std(low_centered, 0, 1) + eps_val;
            low_norm = low_centered ./ low_sd;

            % 5) high residual은 독립 normalize하지 않고 약하게 추가
            high_weak = high ./ low_sd;

            out = low_norm + residual_beta * high_weak;

        case "trend_residual_soft"
            % trend_residual의 soft normalization 버전
            % low trend의 std가 작을 때 과도한 증폭을 방지

            if center_before_savgol
                x_centered = x - mean(x, 1);
            else
                x_centered = x;
            end

            sg_window_valid = make_valid_sg_window( ...
                sg_window, ...
                size(x_centered, 1), ...
                sg_polyorder ...
            );

            low = sgolayfilt(x_centered, sg_polyorder, sg_window_valid);
            high = x_centered - low;

            low_mu = mean(low, 1);
            low_centered = low - low_mu;

            low_sd = std(low_centered, 0, 1);
            denom = sqrt(low_sd.^2 + sig_floor.^2) + eps_val;

            low_norm = low_centered ./ denom;
            high_weak = high ./ denom;

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