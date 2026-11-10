clc; clear;

% ============================================
% Collect Onset / Offset from sentence*_onoff.mat
% ============================================

% 기본 경로
path_cur = cd();
sub_id   = 6;   % 이미 정의되어 있다고 가정

% 읽을 디렉토리
in_dir = fullfile(path_cur, 'data', 'preproc', ...
                  sprintf('sub%d', sub_id), ...
                  'sentence', 'IMU', 'session_1_multi_pick');

% 저장 디렉토리
out_dir = fullfile(path_cur, 'data', 'orig', sprintf('sub%d', sub_id));

% --------------------------------------------
% sentence*_onoff.mat 파일 목록
% --------------------------------------------
files = dir(fullfile(in_dir, 'sentence*_onoff.mat'));
n_sent = length(files);

if n_sent == 0
    error('sentence*_onoff.mat 파일이 없습니다.');
end

% --------------------------------------------
% sentence 번호 추출 후 정렬
% --------------------------------------------
sent_idx = zeros(n_sent,1);

for i = 1:n_sent
    tok = regexp(files(i).name, 'sentence(\d+)_onoff\.mat', 'tokens');
    sent_idx(i) = str2double(tok{1}{1});
end

[sent_idx_sorted, order] = sort(sent_idx);
files = files(order);

% --------------------------------------------
% 결과 배열 초기화
% (문장 개수) x 6
% --------------------------------------------
Onset   = zeros(n_sent, 6);
Offset  = zeros(n_sent, 6);

% --------------------------------------------
% 파일 하나씩 로드
% --------------------------------------------
for i = 1:n_sent
    file_path = fullfile(in_dir, files(i).name);
    S = load(file_path);

    if ~isfield(S, 'Onset') || ~isfield(S, 'Offset')
        error('파일 %s 에 Onset 또는 Offset 변수가 없습니다.', files(i).name);
    end

    Onset(i, :)  = S.Onset(:).';
    Offset(i, :) = S.Offset(:).';
end

% --------------------------------------------
% 저장
% --------------------------------------------
save_path = fullfile(out_dir, 'Triggers.mat');
save(save_path, 'Onset', 'Offset');

fprintf('Triggers.mat 저장 완료 (%d 문장)\n', n_sent);
