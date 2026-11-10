clc; clear;

FOLDER_NAME = 'session_1';
SUB_ID = 6;
% 전체 문장 할 때는 []
% 한 문장 할 때는 문장 번호
% 여러 문장 할 때는 [3 7 12] / 1:10
SENTENCE_LIST = 3;

pick_multi_onoff_sentences(SUB_ID, FOLDER_NAME, SENTENCE_LIST);

function pick_multi_onoff_sentences(sub_id, FOLDER_NAME, sentence_list)
% pick_multi_onoff_sentences(sub_id)
% - sub_id: 예) 6, 7 ...

    % =========================
    % 입력 체크
    % =========================
    if nargin < 3 || isempty(sentence_list)
    sentence_list = 1:30;   % 기본: 전체 문장
    end
    if nargin < 1 || isempty(sub_id)
        error('sub_id를 넣어줘. 예: pick_multi_onoff_sentences(6)');
    end
    if ~isscalar(sub_id) || ~isnumeric(sub_id)
        error('sub_id는 숫자 스칼라여야 함. 예: 6');
    end

    % =========================
    % Path 설정
    % =========================
    path_cur = cd();
    path_load_imu = fullfile(path_cur, 'data', 'orig', ...
                             sprintf('sub%d', sub_id), ...
                             'sentence', 'IMU', FOLDER_NAME);

    % 문장 단어 리스트 CSV (1~30문장)
    sentence_csv_path = fullfile(path_cur, 'utils', 'final_sentence_list.csv');
    % 문장 내 단어 단위로 달라서 
    sentence = readtable(sentence_csv_path, 'ReadVariableNames', false);
    strData = string(sentence{:, :});   % (SenNum x ?)
    clear sentence

    SenNum  = 30;
    MaxWord = 6;

    % 문장별 단어 수 계산 (빈 칸 제외)
    WordNum = zeros(SenNum, 1);
    for i_sen = sentence_list
        WordNum(i_sen) = nnz(strlength(strData(i_sen, :)) > 0);
    end

    % 저장 폴더
    out_dir = fullfile(path_cur, 'data', 'preproc', sprintf('sub%d', sub_id), ...
                       'sentence', 'IMU', 'session_1_multi_pick');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    % 전체 요약 저장용 (SenNum x MaxWord)
    Onset_all  = nan(SenNum, MaxWord);
    Offset_all = nan(SenNum, MaxWord);
    T_list     = nan(SenNum, 1);

    % 폴더 존재 체크
    if ~exist(path_load_imu, 'dir')
        error('IMU 폴더가 존재하지 않음: %s', path_load_imu);
    end

    cd(path_load_imu);

    for i_sen = sentence_list

        % 이 문장의 단어 수(최대 6으로 제한)
        MaxWordThis = min(MaxWord, WordNum(i_sen));
        if MaxWordThis <= 0
            Onset = nan(1, MaxWord);
            Offset = nan(1, MaxWord);
            Onset_all(i_sen, :) = Onset;
            Offset_all(i_sen, :) = Offset;
            save(fullfile(out_dir, sprintf('sentence%d_onoff.mat', i_sen)), ...
                 'Onset', 'Offset');
            continue;
        end

        % ---------- 데이터 로드 & 30축 구성 ----------
        [tmp_imu_30xT, T] = load_and_preproc_30xT(i_sen);
        T_list(i_sen) = T;

        % ---------- Plot (10x3) ----------
        hFig = figure(100); clf;
        set(hFig, 'Color', 'w', 'Name', sprintf('sub%d | Sentence %d', sub_id, i_sen), ...
                  'KeyPressFcn', @onKeyPress);

        tiledlayout(10,3, 'TileSpacing','compact', 'Padding','compact');

        ax = gobjects(30,1);
        for sensor = 1:10
            for axis_i = 1:3
                k = (sensor-1)*3 + axis_i; % 1..30
                ax(k) = nexttile;
                plot(1:T, tmp_imu_30xT(k,:), 'LineWidth', 0.8);
                grid on;

                if axis_i == 1
                    ylabel(sprintf('S%d', sensor));
                end
                if sensor < 10
                    set(gca, 'XTickLabel', []);
                else
                    xlabel('Sample');
                end
            end
        end
        linkaxes(ax, 'x');

        sgtitle(sprintf(['sub%d | Sentence %d (Words=%d) | Left click: onset/offset | ', ...
                         'Backspace: undo | r: reset | n: next/save | q: quit/save'], ...
                         sub_id, i_sen, MaxWordThis), ...
                'FontWeight', 'bold');

        % ---------- 상태 변수 ----------
        Onset  = nan(1, MaxWord);
        Offset = nan(1, MaxWord);

        cur_word  = 1;            % 1..MaxWordThis
        expecting = "onset";      % "onset" -> "offset" -> next word

        % undo용 히스토리: [word_idx, type(1=onset/2=offset), value]
        hist = zeros(0,3);

        done_next = false;
        done_quit = false;

        % 단어별 색
        word_colors = lines(MaxWord);

        set(hFig, 'WindowButtonDownFcn', @onMouseClick);

        % ---------- 사용자 조작 루프 ----------
        while isvalid(hFig) && ~done_next && ~done_quit
            drawnow;
            pause(0.05);
        end

        if ~isvalid(hFig)
            done_quit = true;
        end

        % ---------- 현재 문장 결과 저장 ----------
        Onset_all(i_sen, :)  = Onset;
        Offset_all(i_sen, :) = Offset;

        save(fullfile(out_dir, sprintf('sentence%d_onoff.mat', i_sen)), ...
             'Onset', 'Offset', 'T', 'MaxWordThis', 'sub_id');

        if isvalid(hFig)
            close(hFig);
        end

        if done_quit
            break;
        end
    end

    % ---------- 전체 요약 저장 ----------
    cd(path_cur);
    save(fullfile(out_dir, 'ALL_onoff_summary.mat'), ...
         'Onset_all', 'Offset_all', 'T_list', 'SenNum', 'MaxWord', 'WordNum', 'sub_id');

    fprintf('Saved to: %s\n', out_dir);

    % ==========================================================
    % Nested callback functions
    % ==========================================================
    function onMouseClick(~, ~)
        clickType = get(hFig, 'SelectionType');
        if ~strcmp(clickType, 'normal')
            return;
        end

        if cur_word > MaxWordThis
            return;
        end

        cp = get(gca, 'CurrentPoint');
        idx = round(cp(1,1));
        idx = max(1, min(T, idx));

        if expecting == "onset"
            Onset(cur_word) = idx;
            addVLine(idx, cur_word, 1);
            hist(end+1,:) = [cur_word, 1, idx];
            expecting = "offset";
        else
            if ~isnan(Onset(cur_word)) && idx < Onset(cur_word)
                tmp = Onset(cur_word);
                Onset(cur_word) = idx;
                idx = tmp;
                redrawAllLines();
            end

            Offset(cur_word) = idx;
            addVLine(idx, cur_word, 2);
            hist(end+1,:) = [cur_word, 2, idx];

            cur_word = cur_word + 1;
            expecting = "onset";
        end

        updateTitleStatus();
    end

    function onKeyPress(~, event)
        key = lower(event.Key);
        switch key
            case 'n'
                done_next = true;
            case 'q'
                done_quit = true;
            case 'r'
                resetAll();
            case {'backspace','delete'}
                undoLast();
        end
    end

    function addVLine(idx, w, type)
        col = word_colors(w,:);
        if type == 1
            ls = '--'; lw = 1.6;
        else
            ls = '-';  lw = 2.0;
        end
        for a = 1:numel(ax)
            axes(ax(a)); %#ok<LAXES>
            xline(idx, 'LineStyle', ls, 'LineWidth', lw, 'Color', col, 'Tag', 'ONOFF_LINE');
        end
    end

    function redrawAllLines()
        for a = 1:numel(ax)
            delete(findall(ax(a), 'Type', 'ConstantLine', 'Tag', 'ONOFF_LINE'));
        end
        for w = 1:MaxWordThis
            if ~isnan(Onset(w)),  drawOne(w, 1, Onset(w));  end
            if ~isnan(Offset(w)), drawOne(w, 2, Offset(w)); end
        end
        drawnow;
    end

    function drawOne(w, type, idx)
        col = word_colors(w,:);
        if type == 1
            ls = '--'; lw = 1.6;
        else
            ls = '-';  lw = 2.0;
        end
        for a = 1:numel(ax)
            axes(ax(a)); %#ok<LAXES>
            xline(idx, 'LineStyle', ls, 'LineWidth', lw, 'Color', col, 'Tag', 'ONOFF_LINE');
        end
    end

    function undoLast()
        if isempty(hist), return; end
        last = hist(end,:); hist(end,:) = [];
        w = last(1); type = last(2);

        if type == 1
            Onset(w) = nan;
            expecting = "onset";
            cur_word = min(cur_word, w);
        else
            Offset(w) = nan;
            expecting = "offset";
            cur_word = w;
        end

        redrawAllLines();
        updateTitleStatus();
    end

    function resetAll()
        Onset(:)  = nan;
        Offset(:) = nan;
        hist = zeros(0,3);
        cur_word = 1;
        expecting = "onset";
        redrawAllLines();
        updateTitleStatus();
    end

    function updateTitleStatus()
        if cur_word <= MaxWordThis
            msg = sprintf('Now: word %d/%d | expecting %s', cur_word, MaxWordThis, expecting);
        else
            msg = sprintf('Done (%d words). Press n to save/next.', MaxWordThis);
        end
        sgtitle(sprintf('sub%d | Sentence %d (Words=%d) | %s', sub_id, i_sen, MaxWordThis, msg), ...
                'FontWeight', 'bold');
    end

end


% ==========================================================
% Helper: load + preproc -> (30 x T)
% ==========================================================
function [tmp_imu_30xT, T] = load_and_preproc_30xT(i_sen)

    f1 = sprintf('word%d_1.mat', i_sen);
    f2 = sprintf('word%d_2.mat', i_sen);

    S1 = load(f1);
    S2 = load(f2);

    save_data1 = S1.save_data1;
    save_data2 = S2.save_data2;

    savedata1 = save_data1 / 16384;
    savedata2 = save_data2 / 16384;

    len1 = size(savedata1, 3);
    len2 = size(savedata2, 3);
    min_len = min(len1, len2);

    savedata1 = savedata1(:,:,1:min_len);
    savedata2 = savedata2(:,:,1:min_len);

    s = cell(10,1);
    s{1}  = savedata1(1,1:3,:);
    s{2}  = savedata1(2,1:3,:);
    s{3}  = savedata1(3,1:3,:);
    s{4}  = savedata1(4,1:3,:);
    s{5}  = savedata1(5,1:3,:);
    s{6}  = savedata2(1,1:3,:);
    s{7}  = savedata2(2,1:3,:);
    s{8}  = savedata2(3,1:3,:);
    s{9}  = savedata2(4,1:3,:);
    s{10} = savedata2(5,1:3,:);

    tmp_imu_30xT = squeeze([s{1} s{2} s{3} s{4} s{5} s{6} s{7} s{8} s{9} s{10}]);

    tmp_imu_30xT = normalize(tmp_imu_30xT, 2);
    T = size(tmp_imu_30xT, 2);
end
