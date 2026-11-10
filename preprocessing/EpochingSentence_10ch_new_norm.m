clear; close all; clc

path_cur = cd();
path_load = strcat(path_cur, '/data/orig');

% Load Sentence File
sentence_csv_path = strcat(path_cur, '/utils/ver3/final_sentence_list.csv');
sentence = readtable(sentence_csv_path, 'ReadVariableNames', false);
strData = string(sentence{:, :});
strData = lower(strData);
% Load Word File
word_csv_path = strcat(path_cur, '/utils/ver3/final_word_list.csv');
word = readtable(word_csv_path, 'ReadVariableNames', false);
wordData = string(word{:, :});
wordData = lower(wordData);
clear sentence word

% Params
N_ch = 30;
N_sen = 39;

i_sub = 24;

L = 300;

%% ============================================================
% Preprocessing settings
% =============================================================
% "raw"
% "trial_zscore"
% "trial_soft_zscore"
% "trend_residual"
% "trend_residual_soft"
preprocess_mode = "trial_soft_zscore";
% preprocess_mode = "trend_residual";
% preprocess_mode = "trend_residual_soft";

% soft z-score 설정
sig_floor = 0.01;

% Savitzky-Golay trend/residual 설정
sg_window = 10;
sg_polyorder = 4;
residual_beta = 0.3;
center_before_savgol = true;

% padding 후 normalize할 때,
% mean/std 계산을 실제 신호 구간만 사용할지 여부
% true 추천
use_valid_stats_only = true;

eps_val = 1e-8;

cd(path_load)
cd(strcat('sub', string(i_sub)))
load("Triggers.mat")

% Data array, rm NaN after preprocessing
sig_pad = NaN(N_ch, L, 500); % Padded signal
sig_mask = false(L, 500); % Binary for padding or not
pad_idx = false(500, 1); % Signal length w/o padding -> Trueeeee
session_nums = NaN(500, 1); % Number of set (session?)
sentence_nums = NaN(500, 1); % Number of sentence
word_nums = NaN(500, 1); % Number of word
sig_len = NaN(500, 1); % Signal length w/o padding
max_len = NaN(500, 1); % Max length of signal -> fix to "300"?


i = 1;
for i_sen = 1:N_sen
    % %% For window
    % load(strcat('sentence\IMU\session_1\word', string(i_sen), '.mat'))

    %% For mac
    load(strcat('sentence/IMU/session_1//word', string(i_sen), '_1.mat'))
    load(strcat('sentence/IMU/session_1//word', string(i_sen), '_2.mat'))

    len1 = size(save_data1, 3);
    len2 = size(save_data2, 3);
    min_len = min(len1, len2);

    save_data1 = save_data1(:,:,1:min_len);
    save_data2 = save_data2(:,:,1:min_len);

    % Extract IMU data (1:3 = IMU (16384), 4:6 = Gyro (131))
    savedata_s1 = save_data1(1, 1:3, :);
    savedata_s2 = save_data1(2, 1:3, :);
    savedata_s3 = save_data1(3, 1:3, :);
    savedata_s4 = save_data1(4, 1:3, :);
    savedata_s5 = save_data1(5, 1:3, :);
    savedata_s6 = save_data2(1, 1:3, :);
    savedata_s7 = save_data2(2, 1:3, :);
    savedata_s8 = save_data2(3, 1:3, :);
    savedata_s9 = save_data2(4, 1:3, :);
    savedata_s10 = save_data2(5, 1:3, :);
    tmp_imu = squeeze([savedata_s1 savedata_s2 savedata_s3 savedata_s4 savedata_s5 ...
                       savedata_s6 savedata_s7 savedata_s8 savedata_s9 savedata_s10])/16384;

    % 3-sample smoothing (time dimension)
    % tmp_imu = movmean(tmp_imu, 3, 2);

    clear savedata*

    % Count # of words in sentence
    LenStr = nnz(strlength(strData(i_sen,:))>0);

    for i_word = 1:LenStr
        tmp_data = tmp_imu(:, Onset(i_sen, i_word):Offset(i_sen, i_word));
        % tmp_data = normalize(tmp_data,2);

        % len_word = length(tmp_data);
        % if len_word > L
        %     sig_pad(:, :, i) = tmp_data(:,1:L);
        % else    
        %     sig_pad(:, :, i) = padding_data(tmp_data, L);
        % end

        % 길이는 time dimension 기준으로 계산
        len_word = size(tmp_data, 2);
        
        % 1) 먼저 시간축 길이를 L=300으로 맞춤
        if len_word > L
            tmp_padded = tmp_data(:, 1:L);
            valid_len = L;
        else
            tmp_padded = padding_data(tmp_data, L);
            valid_len = len_word;
        end
        
        % 2) 길이를 300으로 맞춘 뒤 전처리/normalize 적용
        tmp_padded = preprocess_padded_segment( ...
            tmp_padded, ...
            valid_len, ...
            preprocess_mode, ...
            sig_floor, ...
            sg_window, ...
            sg_polyorder, ...
            residual_beta, ...
            center_before_savgol, ...
            use_valid_stats_only, ...
            eps_val ...
        );
        
        % 3) 저장
        sig_pad(:, :, i) = tmp_padded;
        
        % Preprocess data
        %sig_pad(:, :, i) = padding_data(tmp_data, L);
        % if len_word > L
        %     sig_mask(1:L, i) = true;
        % else
        %     sig_mask(1:len_word, i) = true;
        % end
        if valid_len >= L
            sig_mask(1:L, i) = true;
        else
            sig_mask(1:valid_len, i) = true;
        end
        pad_idx(i) = true;
        set_nums(i) = ceil(i_sen/10);
        sentence_nums(i) = i_sen;
        idx = find(wordData == strData(i_sen, i_word), 1);
        if isempty(idx)
            word_nums(i) = 0;
        else
            word_nums(i) = idx-1;
        end
        %word_nums(i) = mod(find(wordData == strData(i_sen, i_word)), 100);
        %word_nums(i) = find(wordData == lower(strData(i_sen, i_word)));
        sig_len(i) = len_word;
        max_len(i) = 300;
        i = i + 1;
    end
end

NumWords = ~isnan(sig_pad(1, 1, :));

sig_pad = sig_pad(:, :, NumWords);
sig_mask = sig_mask(:, NumWords);
pad_idx = pad_idx(NumWords);
set_nums = set_nums(NumWords);
sentence_nums = sentence_nums(NumWords);
word_nums = word_nums(NumWords);
sig_len = sig_len(NumWords);
max_len = max_len(NumWords);

% ==========================================
% [추가] shape 변경: (C,T,W) -> (W,T,C)
% ==========================================
sig_pad = permute(sig_pad, [3 2 1]);   % (단어, 시간, 채널)

% (선택) mask도 같이 (T,W) -> (W,T) 로
sig_mask = sig_mask.';                % (단어, 시간)

cd('../../')
cd('feat')
cd(strcat('sub', string(i_sub)))

save(strcat('sub', string(i_sub), '_sentence_', preprocess_mode, '.mat'), ...
    "sig_pad", ...
    "sig_mask", ...
    "pad_idx", ...
    "set_nums", ...
    "sentence_nums", ...
    "word_nums", ...
    "sig_len", ...
    "max_len", ...
    "preprocess_mode", ...
    "sig_floor", ...
    "sg_window", ...
    "sg_polyorder", ...
    "residual_beta", ...
    "center_before_savgol", ...
    "use_valid_stats_only" ...
)

function output = zero_padding_data(input, len)
    ch = size(input,1);

    tmp = zeros(ch, len);
    tmp(:,1:size(input,2)) = input;

    output = tmp;
end


function output = padding_data(input, len)
    % input: (ch x T)
    % len  : target length
    
    [ch, T] = size(input);

    % 1) 기본 출력 크기
    output = zeros(ch, len);

    % 2) 원본은 그대로 복사
    output(:, 1:T) = input;

    % 3) 이미 길이가 len 이상이면 자를 일은 바깥에서 처리하지만,
    %    안전하게 여기서도 처리 (혹시나 들어오면)
    if T >= len
        output = input(:, 1:len);
        return;
    end

    % 4) padding 구간을 "마지막 10 samples"의 mean/std 기반 Gaussian noise로 채우기
    win = min(10, T);                % T가 10보다 짧을 수도 있으니
    ref = input(:, T-win+1:T);       % (ch x win)

    mu = mean(ref, 2);               % (ch x 1)
    sigma = std(ref, 0, 2);          % (ch x 1)  (N-1 정규화)

    % sigma가 너무 작으면 노이즈가 거의 0이 되므로 최소값 보장
    sigma_floor = 1e-6;
    sigma = max(sigma, sigma_floor);

    pad_len = len - T;               % 채워야 할 길이

    % 채널별로 mu/sigma 적용된 가우시안 노이즈 생성
    % randn(ch, pad_len)은 표준정규 -> sigma 곱하고 mu 더함
    noise = mu + sigma .* randn(ch, pad_len);

    output(:, T+1:len) = noise;
end

function out = preprocess_padded_segment( ...
    x, ...
    valid_len, ...
    preprocess_mode, ...
    sig_floor, ...
    sg_window, ...
    sg_polyorder, ...
    residual_beta, ...
    center_before_savgol, ...
    use_valid_stats_only, ...
    eps_val ...
)
    % x: (C x L), 예: (30 x 300)
    % valid_len: padding 제외 실제 신호 길이
    %
    % 각 channel = sensor/axis
    % time dimension = 2번째 차원

    x = double(x);
    L = size(x, 2);

    valid_len = min(valid_len, L);

    if use_valid_stats_only
        stat_idx = 1:valid_len;
    else
        stat_idx = 1:L;
    end

    switch string(preprocess_mode)

        case "raw"
            out = x;

        case "trial_zscore"
            % 기존 normalize(tmp_data,2)와 유사
            % 단, 길이를 300으로 맞춘 뒤 적용
            mu = mean(x(:, stat_idx), 2);
            sd = std(x(:, stat_idx), 0, 2) + eps_val;

            out = (x - mu) ./ sd;

        case "trial_soft_zscore"
            % soft z-score
            % 기존:
            %   (x - mean) / std
            %
            % 수정:
            %   (x - mean) / sqrt(std^2 + sig_floor^2)
            %
            % 움직임이 거의 없는 channel이 과도하게 커지는 것을 방지

            mu = mean(x(:, stat_idx), 2);
            x_centered = x - mu;

            sd = std(x_centered(:, stat_idx), 0, 2);
            denom = sqrt(sd.^2 + sig_floor.^2) + eps_val;

            out = x_centered ./ denom;

        case "trend_residual"
            % 1) sensor/axis-wise 평균 제거
            if center_before_savgol
                mu = mean(x(:, stat_idx), 2);
                x_centered = x - mu;
            else
                x_centered = x;
            end

            % 2) SG filter로 low-frequency articulatory trend 추출
            sg_window_valid = make_valid_sg_window( ...
                sg_window, ...
                size(x_centered, 2), ...
                sg_polyorder ...
            );

            % x는 (C x T)이므로 transpose해서 시간축 방향 filtering
            low = sgolayfilt(x_centered', sg_polyorder, sg_window_valid)';
            
            % 3) high-frequency residual
            high = x_centered - low;

            % 4) low trend만 sensor/axis-wise z-normalize
            low_mu = mean(low(:, stat_idx), 2);
            low_centered = low - low_mu;

            low_sd = std(low_centered(:, stat_idx), 0, 2) + eps_val;

            low_norm = low_centered ./ low_sd;

            % 5) high residual은 독립 normalize하지 않고 low_sd 기준으로 약하게 추가
            high_weak = high ./ low_sd;

            out = low_norm + residual_beta * high_weak;

        case "trend_residual_soft"
            % trend_residual의 soft normalization 버전
            % low trend std가 작을 때 과도한 증폭 방지

            if center_before_savgol
                mu = mean(x(:, stat_idx), 2);
                x_centered = x - mu;
            else
                x_centered = x;
            end

            sg_window_valid = make_valid_sg_window( ...
                sg_window, ...
                size(x_centered, 2), ...
                sg_polyorder ...
            );

            low = sgolayfilt(x_centered', sg_polyorder, sg_window_valid)';
            high = x_centered - low;

            low_mu = mean(low(:, stat_idx), 2);
            low_centered = low - low_mu;

            low_sd = std(low_centered(:, stat_idx), 0, 2);
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