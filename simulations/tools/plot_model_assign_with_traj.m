function return_info = plot_model_assign_with_traj(X_train_all,model_assign_table,encoder_type_list,plot_info)
    return_info = [];
    model_assign_acc_mean = 0;
    
    if_plot_all_model = 1;
    if_plot_each_model = 0;
    subplot_row_num = 5;
    
    c = color_library();
    %     Traj_test_color = c(42,:);
%         model_colors = c([1,2,2,1,3],:);
    model_colors = c([1:40],:);

    X_train_all = plot_info.train_vel;
    arrange_type = 'sort';
%     traj_type = 'cumsumed';
%     traj_type = 'decoded';

%     kin_type = 'PVA';
    kin_type = plot_info.kin_type;
    traj_type = plot_info.traj_type;
    err_smooth_winsize = plot_info.err_smooth_winsize;
    plot_iter_no = plot_info.plot_iter_no;
    keep_percent = plot_info.keep_percent;
    min_total_err = plot_info.min_total_err;
    short_date_session = plot_info.short_date_session;
    train_state = plot_info.train_state;
    X_train_stroke = X_train_all(:,logical(train_state));

    unique_char_list = plot_info.unique_char_list;
    trial_target_list = plot_info.trial_target_list;
    trial_target_length_list = plot_info.trial_target_length_list;
    trial_target_stroke_length_list = plot_info.trial_target_stroke_length_list;
    train_trial_index = plot_info.train_trial_index;

    train_trial_num = numel(train_trial_index);
    %%% pack_train_trial_index 这是训练集的默认顺序的trial index
    pack_train_trial_index = 1:train_trial_num;
    if strcmp(arrange_type,'unsort')
        train_trial_index_plot = pack_train_trial_index;
    elseif strcmp(arrange_type,'sort')
        %%% 要按照字sort的话，先把训练集中每一个trial对应的字的target找出来，
        % 然后根据字的target排序，把排序得到的index用来重新排列训练集中这些trial的顺序
        % 得到的train_trial_index_plot，表示的是画图的时候，训练集的这些trial怎么排
        % 所以在画图的时候，要得到训练集的信息
        train_trial_target = trial_target_list(train_trial_index);
        [~,sort_train_trial_target_index] = sort(train_trial_target);
        train_trial_index_plot = pack_train_trial_index(sort_train_trial_target_index);
    end


    %%% 计算要画的训练集的trial的mask
    % 得到训练集的信息，包括：训练集所有trial的每一个trial的长度、训练集的每一个字的target、训练集的每一个字的head和tail
    train_trial_target_list = trial_target_list(train_trial_index);
    train_trial_target_length_list = trial_target_length_list(train_trial_index);
    train_head_tail_info = [];
    train_trial_target_length_list_cumsum = cumsum(train_trial_target_length_list);
    for train_trial_no = 1:train_trial_num
        if train_trial_no == 1
            train_head_tail_info{train_trial_no}.head = 1;
        else
            train_head_tail_info{train_trial_no}.head = train_trial_target_length_list_cumsum(train_trial_no-1) + 1;
        end
        train_head_tail_info{train_trial_no}.tail = train_trial_target_length_list_cumsum(train_trial_no);
    end

    %%% 得到光是笔画的head和tail
    train_trial_target_stroke_length_list = trial_target_stroke_length_list(train_trial_index);
    train_stroke_head_tail_info = [];
    train_trial_target_stroke_length_list_cumsum = cumsum(train_trial_target_stroke_length_list);
    for train_trial_no = 1:train_trial_num
        if train_trial_no == 1
            train_stroke_head_tail_info{train_trial_no}.head = 1;
        else
            train_stroke_head_tail_info{train_trial_no}.head = train_trial_target_stroke_length_list_cumsum(train_trial_no-1) + 1;
        end
        train_stroke_head_tail_info{train_trial_no}.tail = train_trial_target_stroke_length_list_cumsum(train_trial_no);
    end

    tmp_model_list = encoder_type_list;
%     tmp_model_num = numel(find(tmp_model_list ~= 0));

    tmp_model_assign_table = model_assign_table;

    [~,tmp_select_model] = max(tmp_model_assign_table,[],1);
    
    tmp_model_num = numel(unique(tmp_select_model));
    %% sort model color
    

    %%% according to select data compare with ground truth
    if plot_info.params.if_replace_true_data_with_simulated
        if plot_info.model_assign_table_GT
            model_colors_sort = model_colors;
            model_assign_acc = 1;
        else
            assign_table_GT = plot_info.assign_table_GT;
            most_consist_context = [];
            model_assign_acc = [];
            for tmp_model_no = 1:tmp_model_num
                tmp_model_assign_table = model_assign_table(ones(1,tmp_model_num)*tmp_model_no,:);
                consist_assign_table = tmp_model_assign_table + assign_table_GT;
                tmp_consist_num = sum(consist_assign_table==2,2);
                [tmp_most_consist_context_num,tmp_most_consist_context] = max(tmp_consist_num);
                most_consist_context = [most_consist_context tmp_most_consist_context];
                model_colors_sort = model_colors(most_consist_context,:);
                tmp_model_assign_acc = tmp_most_consist_context_num./sum(model_assign_table(tmp_model_no,:));
                model_assign_acc = [model_assign_acc tmp_model_assign_acc];
            end
        end
        model_assign_acc_mean = mean(model_assign_acc);
        return_info.model_assign_acc = model_assign_acc_mean;

    else
        %%% according to select data size
        model_select_data_size = sum(tmp_model_assign_table,2);
        %%% according to select data direction
        model_select_data_direction = [];
        for tmp_model_no = 1:tmp_model_num
            tmp_vel = X_train_stroke(:,logical(tmp_model_assign_table(tmp_model_no,:)));
            tmp_vel_sum = mean(calAngle(tmp_vel(1,:),tmp_vel(2,:),'up')*pi./180);
            model_select_data_direction = [model_select_data_direction tmp_vel_sum];
        end
        %%% according to select data direction and data size
        model_select_data_size_direction = mapminmax(model_select_data_size,0,1) + mapminmax(model_select_data_direction,0,1);
    
    %     model_select_data = model_select_data_size;
        model_select_data = model_select_data_direction;
    %     model_select_data = model_select_data_size_direction;
    
        [~,model_select_data_sort_index] = sort(model_select_data,'descend');
        [~,model_color_sort_index,~] = unique(model_select_data_sort_index);
        model_colors_sort = model_colors(model_color_sort_index,:);

    end

    %%
    if if_plot_all_model
        figure();
        if plot_info.if_in_E_disk
            set(gcf,'Position',get(0,'ScreenSize'));
        else
%             set(gcf,'Position',[10,10,1500,700]);
            set(gcf,'Position',[1,41,1536,749.6])
        end    

        Traj_test_linewidth = 1;
        Traj_predict_linewidth = 3;
        Traj_train_linewidth = 3;

        %%% 在画图的时候，遍历的是所有训练集的trial
        % train_trial_index_plot 代表的是在所有训练集中的trial的index （比如训练集只有60个trial）
        % 而train_trial_target 代表的是在所有trial中的训练集的trial index （而总共有90个trial）
        for train_trial_no = 1:train_trial_num
            % select_trial_index 指的是当前要画的是训练集trial的第几个
            select_trial_index = train_trial_index_plot(train_trial_no);
            original_trial_index = train_trial_index(train_trial_no);
            select_trial_target = train_trial_target_list(select_trial_index);
            if plot_info.if_label_type_is_cell
                select_char_label = unique_char_list{1,select_trial_target+1};
            else
                select_char_label = unique_char_list(1,select_trial_target+1);
            end 
            subplot(subplot_row_num,ceil(train_trial_num./subplot_row_num),train_trial_no);
            tmp_train_mask = train_head_tail_info{1,select_trial_index}.head:train_head_tail_info{1,select_trial_index}.tail;
            tmp_train_stroke_mask = train_stroke_head_tail_info{1,select_trial_index}.head:train_stroke_head_tail_info{1,select_trial_index}.tail;

            if strcmp(kin_type,'PVA')   
                tmp_vel_train = X_train_all(3:4,tmp_train_mask);
                if strcmp(traj_type,'cumsumed')
                    tmp_traj_train = cumsum(tmp_vel_train')';
                elseif strcmp(traj_type,'decoded')
                    tmp_traj_train = X_train_all(1:2,tmp_train_mask);
                end
            elseif strcmp(kin_type,'P')
                tmp_traj_train = X_train_all(1:2,tmp_train_mask);
            elseif strcmp(kin_type,'V')
                tmp_vel_train = X_train_all(1:2,tmp_train_mask);
                tmp_traj_train = cumsum(tmp_vel_train')';
            elseif strcmp(kin_type,'VS')
                tmp_vel_train = X_train_all(1:2,tmp_train_mask);
                tmp_traj_train = cumsum(tmp_vel_train')';
                stroke_train = X_train_all(3,tmp_train_mask);
            end
            
            tmp_point_num = size(tmp_traj_train,2);
            if plot_info.if_decode_stroke == 1 %%% 如果只解码笔画
                tmp_train_state = train_state(tmp_train_mask);
                tmp_tmp_select_model = tmp_select_model(tmp_train_stroke_mask);
                plot_point_cnt = 1;
                for tmp_point_no = 1:tmp_point_num-1
                    if tmp_train_state(tmp_point_no) == 1
                            plot(tmp_traj_train(1,tmp_point_no:tmp_point_no+1),...
                            tmp_traj_train(2,tmp_point_no:tmp_point_no+1),...
                            'Color',model_colors_sort(tmp_tmp_select_model(plot_point_cnt),:),...
                            'LineWidth',Traj_train_linewidth);
                        hold on;
                        plot_point_cnt = plot_point_cnt + 1;
                    end
                end
            else
                tmp_tmp_select_model = tmp_select_model(tmp_train_mask);
                for tmp_point_no = 1:tmp_point_num-1
                    if tmp_train_state(tmp_point_no) == 1
                            plot(tmp_traj_train(1,tmp_point_no:tmp_point_no+1),...
                            tmp_traj_train(2,tmp_point_no:tmp_point_no+1),...
                            'Color',model_colors_sort(tmp_tmp_select_model(tmp_point_no),:),...
                            'LineWidth',Traj_train_linewidth);
                        hold on;
                    end
                end
            end
%             scatter(tmp_traj_train(1,1),tmp_traj_train(2,1),50,'r','filled');

            set(gca,'XTick',[],'YTick',[]);
            axis equal;
            axis off;
            box off;
    %         axis square;
            ylabel([select_char_label ' - ' num2str(select_trial_index)],'fontsize',12);
        end

        
        fig_name = [short_date_session '-err' num2str(min_total_err) '-pack' num2str(plot_info.select_pack_index) '-group' num2str(encoder_type_list) '-smooth' num2str(err_smooth_winsize) '-iter' num2str(plot_iter_no) '-keep' num2str(keep_percent) '-sequence' num2str(plot_info.sequence_length) '-type' num2str(plot_info.neuron_type) '-simulationGT' num2str(plot_info.model_assign_table_GT) '-snr' num2str(plot_info.params.simulate_awgn_snr) '-' num2str(model_assign_acc_mean)];
        sgtitle(fig_name);

        save_path = [plot_info.save_path 'model_assignment\'];
        if exist(save_path,'dir')==0
            mkdir(save_path);
        end
        png_fn = fullfile([save_path fig_name '.png']);
        saveas(gcf,png_fn,'png');
    end
    %% plot for each model
    if if_plot_each_model
        c = color_library();
        model_colors = model_colors_sort;
        for tmp_model_no = 1:tmp_model_num
            tmp_model_colors = model_colors;
            tmp_model_colors(setdiff(1:end,tmp_model_no),:) = repmat([0.8,0.8,0.8],tmp_model_num-1,1);
            figure();
            if plot_info.if_in_E_disk
                set(gcf,'Position',get(0,'ScreenSize'));
            else
%                 set(gcf,'Position',[10,10,1500,700]);
                set(gcf,'Position',[1,41,1536,749.6])
            end    
            
            Traj_test_linewidth = 1;
            Traj_predict_linewidth = 3;
            Traj_train_linewidth = 3;

            %%% 在画图的时候，遍历的是所有训练集的trial
            % train_trial_index_plot 代表的是在所有训练集中的trial的index （比如训练集只有60个trial）
            % 而train_trial_target 代表的是在所有trial中的训练集的trial index （而总共有90个trial）
            for train_trial_no = 1:train_trial_num
                % select_trial_index 指的是当前要画的是训练集trial的第几个
                select_trial_index = train_trial_index_plot(train_trial_no);
                original_trial_index = train_trial_index(train_trial_no);
                select_trial_target = train_trial_target_list(select_trial_index);
                if plot_info.if_label_type_is_cell
                    select_char_label = unique_char_list{1,select_trial_target+1};
                else
                    select_char_label = unique_char_list(1,select_trial_target+1);
                end                 
                subplot(subplot_row_num,ceil(train_trial_num./subplot_row_num),train_trial_no);
                tmp_train_mask = train_head_tail_info{1,select_trial_index}.head:train_head_tail_info{1,select_trial_index}.tail;
                tmp_train_stroke_mask = train_stroke_head_tail_info{1,select_trial_index}.head:train_stroke_head_tail_info{1,select_trial_index}.tail;

                if strcmp(kin_type,'PVA')   
                    tmp_vel_train = X_train_all(3:4,tmp_train_mask);
                    if strcmp(traj_type,'cumsumed')
                        tmp_traj_train = cumsum(tmp_vel_train')';
                    elseif strcmp(traj_type,'decoded')
                        tmp_traj_train = X_train_all(1:2,tmp_train_mask);
                    end
                elseif strcmp(kin_type,'P')
                    tmp_traj_train = X_train_all(1:2,tmp_train_mask);
                elseif strcmp(kin_type,'V')
                    tmp_vel_train = X_train_all(1:2,tmp_train_mask);
                    tmp_traj_train = cumsum(tmp_vel_train')';
                elseif strcmp(kin_type,'VS')
                    tmp_vel_train = X_train_all(1:2,tmp_train_mask);
                    tmp_traj_train = cumsum(tmp_vel_train')';
                    stroke_train = X_train_all(3,tmp_train_mask);
                end
                tmp_point_num = size(tmp_traj_train,2);
                if plot_info.if_decode_stroke == 1 %%% 如果只解码笔画
                    tmp_train_state = train_state(tmp_train_mask);
                    tmp_tmp_select_model = tmp_select_model(tmp_train_stroke_mask);
                    plot_point_cnt = 1;
                    for tmp_point_no = 1:tmp_point_num-1
                        if tmp_train_state(tmp_point_no) == 1
                                plot(tmp_traj_train(1,tmp_point_no:tmp_point_no+1),...
                                tmp_traj_train(2,tmp_point_no:tmp_point_no+1),...
                                'Color',tmp_model_colors(tmp_tmp_select_model(plot_point_cnt),:),...
                                'LineWidth',Traj_train_linewidth);
                            hold on;
                            plot_point_cnt = plot_point_cnt + 1;
                        end
                    end
                else
                    tmp_tmp_select_model = tmp_select_model(tmp_train_mask);
                    for tmp_point_no = 1:tmp_point_num-1
                        if tmp_train_state(tmp_point_no) == 1
                                plot(tmp_traj_train(1,tmp_point_no:tmp_point_no+1),...
                                tmp_traj_train(2,tmp_point_no:tmp_point_no+1),...
                                'Color',tmp_model_colors(tmp_tmp_select_model(tmp_point_no),:),...
                                'LineWidth',Traj_train_linewidth);
                            hold on;
                        end
                    end
                end

%                 scatter(tmp_traj_train(1,1),tmp_traj_train(2,1),50,'r','filled');

                set(gca,'XTick',[],'YTick',[]);
                axis equal;
        %         axis square;
                axis off;
                box off;
                ylabel([select_char_label ' - ' num2str(select_trial_index)],'fontsize',12);
            end

            fig_name = [short_date_session '-err' num2str(min_total_err) '-pack' num2str(plot_info.select_pack_index) '-group' num2str(encoder_type_list) '-smooth' num2str(err_smooth_winsize) '-iter' num2str(plot_iter_no) '-keep' num2str(keep_percent)  '-sequence' num2str(plot_info.sequence_length) '-model' num2str(tmp_model_no) '-type' num2str(plot_info.neuron_type) '-simulationGT' num2str(plot_info.model_assign_table_GT) '-snr' num2str(plot_info.params.simulate_awgn_snr) '-' num2str(model_assign_acc_mean)];
            sgtitle(fig_name);

            save_path = [plot_info.save_path 'model_assignment\'];
            if exist(save_path,'dir')==0
                mkdir(save_path);
            end
            png_fn = fullfile([save_path fig_name '.png']);
            saveas(gcf,png_fn,'png');
        end
    end


end

