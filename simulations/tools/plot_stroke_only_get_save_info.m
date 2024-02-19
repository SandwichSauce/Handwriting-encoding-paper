function decoded_data = plot_stroke_only_get_save_info(train_test_package,arrange_type)

    % 用 真实位置 还是 绝对位置 还是 相对位置
    use_position_type = 'true';
%     use_position_type = 'absolute';
%     use_position_type = 'offset';
%     use_position_type = 'cumsumed';

    if_plot_predict_traj_with_GT = 0;
    if_save_fig = 0;

    if_decode_stroke = train_test_package.params.if_decode_stroke;
    if_decode_break = train_test_package.params.if_decode_break;
    
    package_VAF = train_test_package.package_VAF;
    
    decode_pack_list = train_test_package.params.decode_pack_list;
    decoder_type = train_test_package.params.decoder_type;
    trial_target_list = train_test_package.params.trial_target_list;
    unique_char_list = train_test_package.params.unique_char_list;
    trial_target_length_list = train_test_package.params.trial_target_length_list;
    trial_target_stroke_length_list = train_test_package.params.trial_target_stroke_length_list;
    short_date_session = train_test_package.params.short_date_session;
    cur_time_stamp = train_test_package.params.cur_time_stamp;
    save_path = train_test_package.params.save_path;
    kin_type = train_test_package.params.kin_type;
    traj_type = train_test_package.params.traj_type;
    if train_test_package.params.if_sequence_to_one
        sequence_type = ['-sequence' num2str(train_test_package.test_data{1,decode_pack_list(1)}.decoded_info.TF.options.sequence_length)];
    else
        sequence_type = '';
    end

    params = train_test_package.params;
    if isfield(params,'neuron_type_no')
        neuron_type = ['-neuron type ' num2str(params.neuron_type_no) '-' num2str(params.threshold_R2) '-' num2str(params.threshold_FLDA)];
    else
        neuron_type = '';
    end

    %%% 画的时候遵循一个策略：不管这一个pack有没有解码，都画出来
    % 如果没有解码的话，就把X_test当做X_predict，轨迹完全重合肯定不可能是解码的
    pack_num = numel(train_test_package.train_data);
    test_trial_index_all = [];
    decode_trial_index_all = [];
    X_test_all = [];
    X_predict_all = [];
    test_state_all = [];
    

    for pack_no = 1:pack_num
        test_trial_index = train_test_package.test_data{1,pack_no}.test_trial_index;
        test_trial_index_all = [test_trial_index_all test_trial_index];
        X_test = train_test_package.test_data{1,pack_no}.test_PVA;
        test_state = train_test_package.test_data{1,pack_no}.test_state;
        if strcmp(kin_type,'VS')
            state_separate_stroke_break_test = train_test_package.test_data{1,pack_no}.test_state_separate_stroke_break;
        end
        if strcmp(kin_type,'PVA')   
            X_test = X_test;
        elseif strcmp(kin_type,'P')
            X_test = X_test(1:2,:);
        elseif strcmp(kin_type,'V')
            X_test = X_test(3:4,:);
        elseif strcmp(kin_type,'VS')
            X_test = [X_test(3:4,:);state_separate_stroke_break_test];
        end
        X_test_all = [X_test_all X_test];
        test_state_all = [test_state_all test_state];
        if numel(find(decode_pack_list == pack_no)) == 1
            X_predict = train_test_package.test_data{1,pack_no}.decoded_info.X_predict;
            decode_trial_index_all = [decode_trial_index_all test_trial_index];
        else
            X_predict = X_test;
        end
        X_predict_all = [X_predict_all X_predict];
    end
    
    if if_plot_predict_traj_with_GT
        figure();
%     set(gcf,'Position',get(0,'ScreenSize'));
        set(gcf,'Position',[1,41,1536,749.6]);
    end
    subplot_row_num = 8;
    c = color_library();
    Traj_test_color = c(31,:);
%     Traj_predict_color = c(134,:);

%     Traj_predict_color = [222,177,68]./255; % 黄色 KF
    Traj_predict_color = [75,67,124]./255; % 紫色 DyEnsemble

%     Traj_test_color = [71,94,127]./255;


    Traj_test_linewidth = 2;
    Traj_predict_linewidth = 3;

    test_trial_num = numel(test_trial_index_all);
    if strcmp(arrange_type,'unsort')
        test_trial_index_plot = test_trial_index_all;
        sort_test_trial_target_index = 1:test_trial_num;
    elseif strcmp(arrange_type,'sort')
        test_trial_target = trial_target_list(test_trial_index_all);
        [~,sort_test_trial_target_index] = sort(test_trial_target);
        test_trial_index_plot = test_trial_index_all(sort_test_trial_target_index);
    end
    
    
    if strcmp(train_test_package.params.divide_type,'leave-one-repeat-out')
        head_tail_info = [];
        test_trial_target_length_list = trial_target_length_list(test_trial_index_all);
        test_trial_target_length_list_cumsum = cumsum(test_trial_target_length_list);
        for test_trial_no = 1:test_trial_num
            if test_trial_no == 1
                head_tail_info{test_trial_no}.head = 1;
            else
                head_tail_info{test_trial_no}.head = test_trial_target_length_list_cumsum(test_trial_no-1) + 1;
            end
            head_tail_info{test_trial_no}.tail = test_trial_target_length_list_cumsum(test_trial_no);
        end
        
        %%%
        head_tail_info_stroke_only = [];
%         test_trial_target_stroke_length_list = trial_target_stroke_length_list(test_trial_index_plot);
        test_trial_target_stroke_length_list = trial_target_stroke_length_list(test_trial_index_all);
        test_trial_target_stroke_length_list_cumsum = cumsum(test_trial_target_stroke_length_list);
        for test_trial_no = 1:test_trial_num
            if test_trial_no == 1
                head_tail_info_stroke_only{test_trial_no}.head = 1;
            else
                head_tail_info_stroke_only{test_trial_no}.head = test_trial_target_stroke_length_list_cumsum(test_trial_no-1) + 1;
            end
            head_tail_info_stroke_only{test_trial_no}.tail = test_trial_target_stroke_length_list_cumsum(test_trial_no);
        end
    elseif strcmp(train_test_package.params.divide_type,'leave-one-repeat-out-types')
        head_tail_info = [];
        test_trial_target_length_list = trial_target_length_list(test_trial_index_all);
        test_trial_target_length_list_cumsum = cumsum(test_trial_target_length_list);
        for test_trial_no = 1:test_trial_num
            if test_trial_no == 1
                head_tail_info{test_trial_no}.head = 1;
            else
                head_tail_info{test_trial_no}.head = test_trial_target_length_list_cumsum(test_trial_no-1) + 1;
            end
            head_tail_info{test_trial_no}.tail = test_trial_target_length_list_cumsum(test_trial_no);
        end
        
        %%%
        head_tail_info_stroke_only = [];
%         test_trial_target_stroke_length_list = trial_target_stroke_length_list(test_trial_index_plot);
        test_trial_target_stroke_length_list = trial_target_stroke_length_list(test_trial_index_all);
        test_trial_target_stroke_length_list_cumsum = cumsum(test_trial_target_stroke_length_list);
        for test_trial_no = 1:test_trial_num
            if test_trial_no == 1
                head_tail_info_stroke_only{test_trial_no}.head = 1;
            else
                head_tail_info_stroke_only{test_trial_no}.head = test_trial_target_stroke_length_list_cumsum(test_trial_no-1) + 1;
            end
            head_tail_info_stroke_only{test_trial_no}.tail = test_trial_target_stroke_length_list_cumsum(test_trial_no);
        end    
    elseif strcmp(train_test_package.params.divide_type,'leave-one-char-out')
        head_tail_info = [];
        test_trial_target_length_list = trial_target_length_list(test_trial_index_plot);
        test_trial_target_length_list_cumsum = cumsum(test_trial_target_length_list);
        for test_trial_no = 1:test_trial_num
            if test_trial_no == 1
                head_tail_info{test_trial_no}.head = 1;
            else
                head_tail_info{test_trial_no}.head = test_trial_target_length_list_cumsum(test_trial_no-1) + 1;
            end
            head_tail_info{test_trial_no}.tail = test_trial_target_length_list_cumsum(test_trial_no);
        end
        
        %%%
        head_tail_info_stroke_only = [];
        test_trial_target_stroke_length_list = trial_target_stroke_length_list(test_trial_index_plot);
        test_trial_target_stroke_length_list_cumsum = cumsum(test_trial_target_stroke_length_list);
        for test_trial_no = 1:test_trial_num
            if test_trial_no == 1
                head_tail_info_stroke_only{test_trial_no}.head = 1;
            else
                head_tail_info_stroke_only{test_trial_no}.head = test_trial_target_length_list_cumsum(test_trial_no-1) + 1;
            end
            head_tail_info_stroke_only{test_trial_no}.tail = test_trial_target_length_list_cumsum(test_trial_no);
        end
    end
    
    if if_plot_predict_traj_with_GT
        for test_trial_no = 1:test_trial_num
            select_trial_index = test_trial_index_plot(test_trial_no);
%             select_trial_index_plot = test_trial_index_plot(test_trial_no);
            select_trial_index_plot = sort_test_trial_target_index(test_trial_no);
            select_trial_target = trial_target_list(select_trial_index);
            if train_test_package.params.if_label_type_is_cell
                select_char_label = unique_char_list{1,select_trial_target+1};
            else
                select_char_label = unique_char_list(1,select_trial_target+1);
            end
            subplot(subplot_row_num,ceil(test_trial_num./subplot_row_num),test_trial_no);
            tmp_mask = head_tail_info{1,select_trial_index_plot}.head:head_tail_info{1,select_trial_index_plot}.tail;
            tmp_mask_stroke_only = head_tail_info_stroke_only{1,select_trial_index_plot}.head:head_tail_info_stroke_only{1,select_trial_index_plot}.tail;
            tmp_test_state = test_state_all(:,tmp_mask);
            if strcmp(kin_type,'PVA')   
                tmp_vel_test = X_test_all(3:4,tmp_mask);
                if strcmp(traj_type,'cumsumed')
                    tmp_traj_test = cumsum(tmp_vel_test')';
                elseif strcmp(traj_type,'decoded')
                    tmp_traj_test = X_test_all(1:2,tmp_mask);
                end
            elseif strcmp(kin_type,'P')
                tmp_traj_test = X_test_all(:,tmp_mask);
            elseif strcmp(kin_type,'V') || strcmp(kin_type,'VS')
                tmp_vel_test = X_test_all(1:2,tmp_mask);
                tmp_traj_test = cumsum(tmp_vel_test')';
            end
            tmp_point_num = size(tmp_traj_test,2);
            for tmp_point_no = 1:tmp_point_num-1
                if tmp_test_state(tmp_point_no) == 1
                    plot(tmp_traj_test(1,tmp_point_no:tmp_point_no+1),...
                        tmp_traj_test(2,tmp_point_no:tmp_point_no+1),...
                        'Color',Traj_test_color,'LineWidth',Traj_test_linewidth);
                    hold on;
                end
            end
            tmp_traj_predict_original = [];
            if numel(find(decode_trial_index_all == select_trial_index)) == 1
                if strcmp(kin_type,'PVA')   
                    tmp_vel_predict = X_predict_all(3:4,tmp_mask_stroke_only);
                    if strcmp(traj_type,'cumsumed')
                        tmp_traj_predict = cumsum(tmp_vel_predict')';
                    elseif strcmp(traj_type,'decoded')
                        tmp_traj_predict = X_predict_all(1:2,tmp_mask_stroke_only);
                        tmp_traj_predict = smoothdata(tmp_traj_predict,2,'movmean',10);
                    end
                elseif strcmp(kin_type,'P')
                    tmp_traj_predict = X_predict_all(:,tmp_mask_stroke_only);
                    tmp_traj_predict = smoothdata(tmp_traj_predict,2,'movmean',10);
                elseif strcmp(kin_type,'V') || strcmp(kin_type,'VS')
                    tmp_vel_predict = X_predict_all(1:2,tmp_mask_stroke_only);
                    tmp_traj_predict = cumsum(tmp_vel_predict')';
                end
                tmp_point_num_stroke_only = size(tmp_traj_predict,2);
                tmp_cnt = 1;
                for tmp_point_no = 1:tmp_point_num-1
                    if tmp_point_no > 1
                        if tmp_test_state(tmp_point_no) == 1 && tmp_test_state(tmp_point_no-1) == 0
                            tmp_test_position = tmp_traj_test(:,tmp_point_no);
                            tmp_predict_position = tmp_traj_predict(:,tmp_cnt);
                            tmp_position_offset = tmp_predict_position - tmp_test_position;
                            tmp_traj_predict = tmp_traj_predict - tmp_position_offset;
                        end
                    end
                    if tmp_test_state(tmp_point_no) == 1 && tmp_test_state(tmp_point_no+1) == 1
                        plot(tmp_traj_predict(1,tmp_cnt:tmp_cnt+1),...
                            tmp_traj_predict(2,tmp_cnt:tmp_cnt+1),...
                            'Color',Traj_predict_color,'LineWidth',Traj_predict_linewidth);
                        hold on;
                        tmp_traj_predict_original = [tmp_traj_predict_original tmp_traj_predict(:,tmp_cnt)];
                        tmp_cnt = tmp_cnt + 1;
                    elseif tmp_test_state(tmp_point_no) == 1 && tmp_test_state(tmp_point_no+1) == 0
                        tmp_cnt = tmp_cnt + 1;
                    end
                end
            end
            set(gca,'XTick',[],'YTick',[]);
            axis equal;
            box off;
%             axis off;

%             tmp_traj = [tmp_traj_test tmp_traj_predict];
            tmp_traj = [tmp_traj_test];


            x_min = min(tmp_traj(1,:));
            x_max = max(tmp_traj(1,:));
            y_min = min(tmp_traj(2,:));
            y_max = max(tmp_traj(2,:));
            if strcmp(select_char_label,'口')
                edge_size = 35;
            else
                edge_size = 0;
            end
            x_min = x_min - edge_size;
            x_max = x_max + edge_size + 0.001;
            y_min = y_min - edge_size;
            y_max = y_max + edge_size + 0.001;

%             x_min = -100;
%             x_max = 500;
%             y_min = -500;
%             y_max = 100;

            xlim([x_min,x_max]);
            ylim([y_min,y_max]);
    %         axis square;
            ylabel([select_char_label ' - ' num2str(select_trial_index)],'fontsize',12);
        end
        set(gcf,"Renderer","painters");
        
        fig_name = [short_date_session '-T' cur_time_stamp sequence_type '-' traj_type ...
            ' traj-VAF' num2str(package_VAF) '-' decoder_type '-' kin_type ...
            '-' use_position_type num2str(if_decode_stroke) num2str(if_decode_break) neuron_type];
    %     suptitle(fig_name);
        sgtitle(fig_name);
    
        
        if exist(save_path,'dir')==0
            mkdir(save_path);
        end
        png_fn = fullfile([save_path 'linear' num2str(train_test_package.params.linear_model_num) '-nn' num2str(train_test_package.params.nonlinear_model_num) '-sequence' num2str(train_test_package.params.sequence_length) '-traj-GT' neuron_type '.png']);

%         png_fn = fullfile([save_path 'T' cur_time_stamp sequence_type '-' traj_type ...
%             ' traj-VAF' num2str(package_VAF) '-' decoder_type '-' kin_type ...
%             '-' use_position_type num2str(if_decode_stroke) num2str(if_decode_break) '.png']);
        fig_fn = [png_fn(1:end-4) '.fig'];
        saveas(gcf,png_fn,'png');
        if if_save_fig
            saveas(gcf,fig_fn,'fig');
        end
    end
    
    %% 画一下没有ground truth的
    figure();
%     set(gcf,'Position',get(0,'ScreenSize'));
    set(gcf,'Position',[1,41,1536,749.6]);
    decoded_data = [];
    for test_trial_no = 1:test_trial_num
        select_trial_index = test_trial_index_plot(test_trial_no);
%         select_trial_index_plot = test_trial_index_plot(test_trial_no);

        select_trial_index_plot = sort_test_trial_target_index(test_trial_no);
        decoded_data.trial_index_sort_by_target = sort_test_trial_target_index;
        select_trial_target = trial_target_list(select_trial_index);
        decoded_data.trial_data{1,select_trial_index}.target_from_0 = select_trial_target;
        if train_test_package.params.if_label_type_is_cell
            select_char_label = unique_char_list{1,select_trial_target+1};
        else
            select_char_label = unique_char_list(1,select_trial_target+1);
        end
        decoded_data.trial_data{1,select_trial_index}.label = select_char_label;
        subplot(subplot_row_num,ceil(test_trial_num./subplot_row_num),test_trial_no);
        tmp_mask = head_tail_info{1,select_trial_index_plot}.head:head_tail_info{1,select_trial_index_plot}.tail;
        tmp_mask_stroke_only = head_tail_info_stroke_only{1,select_trial_index_plot}.head:head_tail_info_stroke_only{1,select_trial_index_plot}.tail;
        tmp_test_state = test_state_all(:,tmp_mask);
        if strcmp(kin_type,'PVA')   
            tmp_vel_test = X_test_all(3:4,tmp_mask);
            if strcmp(traj_type,'cumsumed')
                tmp_traj_test = cumsum(tmp_vel_test')';
            elseif strcmp(traj_type,'decoded')
                tmp_traj_test = X_test_all(1:2,tmp_mask);
            end
        elseif strcmp(kin_type,'P')
            tmp_vel_test = [0;0];
            tmp_traj_test = X_test_all(:,tmp_mask);
        elseif strcmp(kin_type,'V') || strcmp(kin_type,'VS')
            tmp_vel_test = X_test_all(1:2,tmp_mask);
            tmp_traj_test = cumsum(tmp_vel_test')';
        end
        decoded_data.trial_data{1,select_trial_index}.vel = tmp_vel_test;
        decoded_data.trial_data{1,select_trial_index}.traj = tmp_traj_test;
        decoded_data.trial_data{1,select_trial_index}.state = tmp_test_state;

        
        tmp_point_num = size(tmp_traj_test,2);


        %%% 单独画GT
%         for tmp_point_no = 1:tmp_point_num-1
%             if tmp_test_state(tmp_point_no) == 1
%                 plot(tmp_traj_test(1,tmp_point_no:tmp_point_no+1),...
%                     tmp_traj_test(2,tmp_point_no:tmp_point_no+1),...
%                     'Color',Traj_test_color,'LineWidth',Traj_predict_linewidth);
%                 hold on;
%             end
%         end



        tmp_traj_predict_original = [];
        if numel(find(decode_trial_index_all == select_trial_index)) == 1
            if strcmp(kin_type,'PVA')   
                tmp_vel_predict = X_predict_all(3:4,tmp_mask_stroke_only);
                if strcmp(traj_type,'cumsumed')
                    tmp_traj_predict = cumsum(tmp_vel_predict')';
                elseif strcmp(traj_type,'decoded')
                    tmp_traj_predict = X_predict_all(1:2,tmp_mask_stroke_only);
                    tmp_traj_predict = smoothdata(tmp_traj_predict,2,'movmean',10);
                end
            elseif strcmp(kin_type,'P')
                tmp_vel_predict = [0;0];
                tmp_traj_predict = X_predict_all(:,tmp_mask_stroke_only);
                tmp_traj_predict = smoothdata(tmp_traj_predict,2,'movmean',10);
            elseif strcmp(kin_type,'V') || strcmp(kin_type,'VS')
                tmp_vel_predict = X_predict_all(1:2,tmp_mask_stroke_only);
                tmp_traj_predict = cumsum(tmp_vel_predict')';
            end

            tmp_point_num_stroke_only = size(tmp_traj_predict,2);
            tmp_cnt = 1;
            for tmp_point_no = 1:tmp_point_num-1
                if tmp_point_no > 1
                    if tmp_test_state(tmp_point_no) == 1 && tmp_test_state(tmp_point_no-1) == 0
                        tmp_test_position = tmp_traj_test(:,tmp_point_no);
                        tmp_predict_position = tmp_traj_predict(:,tmp_cnt);
                        tmp_position_offset = tmp_predict_position - tmp_test_position;
                        tmp_traj_predict = tmp_traj_predict - tmp_position_offset;
                    end
                end
                if tmp_test_state(tmp_point_no) == 1 && tmp_test_state(tmp_point_no+1) == 1
                    plot(tmp_traj_predict(1,tmp_cnt:tmp_cnt+1),...
                        tmp_traj_predict(2,tmp_cnt:tmp_cnt+1),...
                        'Color',Traj_predict_color,'LineWidth',Traj_predict_linewidth);
                    hold on;
                    tmp_traj_predict_original = [tmp_traj_predict_original tmp_traj_predict(:,tmp_cnt)];
                    tmp_cnt = tmp_cnt + 1;
                elseif tmp_test_state(tmp_point_no) == 1 && tmp_test_state(tmp_point_no+1) == 0
                    tmp_traj_predict_original = [tmp_traj_predict_original tmp_traj_predict(:,tmp_cnt)];
                    tmp_cnt = tmp_cnt + 1;
                end
            end
            tmp_traj_predict_original = [tmp_traj_predict_original tmp_traj_predict(:,end)];
            decoded_data.trial_data{1,select_trial_index}.vel_predict = tmp_vel_predict;
            decoded_data.trial_data{1,select_trial_index}.traj_predict = tmp_traj_predict_original;

        end
        set(gca,'XTick',[],'YTick',[]);
        axis equal;
        box off;
        axis off;
%         tmp_traj = [tmp_traj_test tmp_traj_predict];
%         tmp_traj = [tmp_traj_test];
        tmp_traj = [tmp_traj_predict_original];


        x_min = min(tmp_traj(1,:));
        x_max = max(tmp_traj(1,:));
        y_min = min(tmp_traj(2,:));
        y_max = max(tmp_traj(2,:));
        if strcmp(select_char_label,'口')
            edge_size = 35;
        else
            edge_size = 0;
        end
        x_min = x_min - edge_size;
        x_max = x_max + edge_size + 0.001;
        y_min = y_min - edge_size;
        y_max = y_max + edge_size + 0.001;

%         x_min = -100;
%         x_max = 500;
%         y_min = -500;
%         y_max = 100;

        xlim([x_min,x_max]);
        ylim([y_min,y_max]);
%         axis square;
%         axis square;
%         ylabel([select_char_label ' - ' num2str(select_trial_index)],'fontsize',12);
    end
    set(gcf,"Renderer","painters");
    
    fig_name = [short_date_session '-T' cur_time_stamp sequence_type '-' traj_type ...
        ' traj-VAF' num2str(package_VAF) '-' decoder_type '-' kin_type ...
        '-' use_position_type num2str(if_decode_stroke) num2str(if_decode_break) '-no GT' neuron_type];
    sgtitle(fig_name);
    
    if exist(save_path,'dir')==0
        mkdir(save_path);
    end
    png_fn = fullfile([save_path 'linear' num2str(train_test_package.params.linear_model_num) '-nn' num2str(train_test_package.params.nonlinear_model_num) '-sequence' num2str(train_test_package.params.sequence_length) '-traj-no GT' neuron_type '.png']);
    fig_fn = [png_fn(1:end-4) '.fig'];
    saveas(gcf,png_fn,'png');
    if if_save_fig
        saveas(gcf,fig_fn,'fig');
    end
    
end