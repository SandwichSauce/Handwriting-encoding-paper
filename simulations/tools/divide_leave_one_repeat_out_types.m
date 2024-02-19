function train_test_package = divide_leave_one_repeat_out_types(trials_data)
    trial_target = trials_data.trial_target;
    trial_data = trials_data.trial_data;
    if_has_val = trials_data.params.if_has_val;
    if_anaylse_stroke = trials_data.params.if_analyse_stroke;
    if_has_inter_trial = trials_data.params.if_has_inter_trial;
    train_test_package = [];
    train_data = [];
    test_data = [];
    
    type_target_index_list = trials_data.type_target_index_list;
%     type_target_index_list{1} = 1:5;
%     type_target_index_list{2} = 6:15;
%     type_target_index_list{3} = 16:27;
%     type_target_index_list{4} = 28:37;
    type_num = numel(type_target_index_list);

    target_repeat_num = unique(trials_data.target_repeat_num_list);
    if numel(target_repeat_num) == 1
        %%% get repeat trial index
        [trial_target_sort,trial_target_sort_index] = sort(trial_target);
        % repeat_trial_index的第X行代表第X遍repeat的trial index 
        repeat_trial_index = reshape(trial_target_sort_index',target_repeat_num,[]);
        
        pack_cnt = 1;
        for type_no = 1:type_num
            type_target_index = type_target_index_list{type_no};
            type_repeat_trial_index = repeat_trial_index(:,type_target_index);


            for target_repeat_no = 1:target_repeat_num
                tmp_train_repeat_trial_index = type_repeat_trial_index;
                if if_has_val
                    tmp_val_repeat_trial_index = type_repeat_trial_index;
                    tmp_test_repeat_no = target_repeat_no;
                    tmp_train_repeat_no_start = mod(tmp_test_repeat_no,target_repeat_num) + 1;
                    tmp_train_repeat_no_end = mod(tmp_test_repeat_no,target_repeat_num) + 1 + target_repeat_num - 2 - 1;
                    if tmp_train_repeat_no_end > target_repeat_num
                        tmp_train_repeat_no_end_actual = tmp_train_repeat_no_end - target_repeat_num;
                        tmp_train_repeat_no_end = target_repeat_num;
                        tmp_train_repeat_no = [1:tmp_train_repeat_no_end_actual tmp_train_repeat_no_start:tmp_train_repeat_no_end];
    
                    else
                        tmp_train_repeat_no_end_actual = tmp_train_repeat_no_end;
                        tmp_train_repeat_no = [tmp_train_repeat_no_start:tmp_train_repeat_no_end];
                    end
                    tmp_val_repeat_no = mod(tmp_train_repeat_no_end_actual,target_repeat_num) + 1;
                    tmp_train_repeat_trial_index([tmp_val_repeat_no,tmp_test_repeat_no],:) = [];
                    tmp_val_repeat_trial_index([tmp_train_repeat_no,tmp_test_repeat_no],:) = [];
                    train_trial_index = sort(tmp_train_repeat_trial_index(:)');
                    val_trial_index = sort(tmp_val_repeat_trial_index(:)');
                else
                    tmp_train_repeat_trial_index(target_repeat_no,:) = [];
                    train_trial_index = sort(tmp_train_repeat_trial_index(:)');
        %             train_trial_index = tmp_repeat_trial_index(:)';
                end
                test_trial_index = sort(type_repeat_trial_index(target_repeat_no,:));
    
                train_data{1,pack_cnt}.train_trial_index = train_trial_index;
                test_data{1,pack_cnt}.test_trial_index = test_trial_index;
                if if_has_val
                    val_data{1,pack_cnt}.val_trial_index = val_trial_index;
                end
                %%% make train data
                train_spike = [];
                train_PVA = [];
                if if_has_inter_trial
                    train_spike_inter_trial = [];
                    train_PVA_inter_trial = [];
                    train_go_mask = [];
                end
                train_stroke_PVA = [];
                train_stroke_spike = [];
                train_break_PVA = [];
                train_break_spike = [];
                train_state = [];
                train_state_unified_stroke_break = [];
                train_state_separate_stroke_break = [];
                train_absolute_position_spike_cell = [];
                train_absolute_position = [];
                train_position_offset_spike_cell = [];
                train_position_offset = [];
                train_position_offset_cumsumed_spike_cell = [];
                train_position_offset_cumsumed_cell = [];
                train_position_offset_cumsumed = [];
                train_stroke_spike_cell = [];
                train_stroke_cell = [];
    
                
    
                for train_trial_no = 1:numel(train_trial_index)
                    tmp_trial_index = train_trial_index(train_trial_no);
                    train_spike = [train_spike trial_data{1,tmp_trial_index}.spike_bin];
                    train_PVA = [train_PVA trial_data{1,tmp_trial_index}.PVA_bin];
                    if if_has_inter_trial
                        train_spike_inter_trial = [train_spike_inter_trial trial_data{1,tmp_trial_index}.spike_bin_inter_trial];
                        train_PVA_inter_trial = [train_PVA_inter_trial trial_data{1,tmp_trial_index}.PVA_bin_inter_trial];
                        train_go_mask = [train_go_mask trial_data{1,tmp_trial_index}.go_mask];
                    end
                    train_stroke_PVA = [train_stroke_PVA trial_data{1,tmp_trial_index}.stroke_PVA];
                    train_stroke_spike = [train_stroke_spike trial_data{1,tmp_trial_index}.stroke_spike];
                    train_break_PVA = [train_break_PVA trial_data{1,tmp_trial_index}.break_PVA];
                    train_break_spike = [train_break_spike trial_data{1,tmp_trial_index}.break_spike];
                    
                    train_absolute_position_spike_cell{train_trial_no} = trial_data{1,tmp_trial_index}.absolute_position_spike_cell;
                    train_absolute_position{train_trial_no} = trial_data{1,tmp_trial_index}.absolute_position;
                    
                    train_position_offset_spike_cell{train_trial_no} = trial_data{1,tmp_trial_index}.position_offset_spike_cell;
                    train_position_offset{train_trial_no} = trial_data{1,tmp_trial_index}.position_offset;
                    train_position_offset_cumsumed_spike_cell{train_trial_no} = trial_data{1,tmp_trial_index}.position_offset_cumsumed_spike_cell;
                    train_position_offset_cumsumed_cell{train_trial_no} = trial_data{1,tmp_trial_index}.position_offset_cumsumed_cell;
                    train_position_offset_cumsumed{train_trial_no} = trial_data{1,tmp_trial_index}.position_offset_cumsumed;
    
                    train_stroke_spike_cell{train_trial_no} = trial_data{1,tmp_trial_index}.stroke_spike_cell;
                    train_stroke_cell{train_trial_no} = trial_data{1,tmp_trial_index}.stroke_cell;
                    
                    train_state = [train_state trial_data{1,tmp_trial_index}.state_bin];
                    if trials_data.params.if_analyse_stroke
                        train_state_unified_stroke_break = [train_state_unified_stroke_break trial_data{1,tmp_trial_index}.state_unified_stroke_break];
                        train_state_separate_stroke_break = [train_state_separate_stroke_break trial_data{1,tmp_trial_index}.state_separate_stroke_break];
                    end
                end
    
                train_data{1,pack_cnt}.train_spike = train_spike;
                train_data{1,pack_cnt}.train_PVA = train_PVA;
                if if_has_inter_trial
                    train_data{1,pack_cnt}.train_spike_inter_trial = train_spike_inter_trial;
                    train_data{1,pack_cnt}.train_PVA_inter_trial = train_PVA_inter_trial;
                    train_data{1,pack_cnt}.train_go_mask = train_go_mask;
                end
                train_data{1,pack_cnt}.train_stroke_PVA = train_stroke_PVA;
                train_data{1,pack_cnt}.train_stroke_spike = train_stroke_spike;
                train_data{1,pack_cnt}.train_break_PVA = train_break_PVA;
                train_data{1,pack_cnt}.train_break_spike = train_break_spike;
                train_data{1,pack_cnt}.train_state = train_state;
                
                train_data{1,pack_cnt}.train_absolute_position_spike_cell = train_absolute_position_spike_cell;
                train_data{1,pack_cnt}.train_absolute_position = train_absolute_position;
                train_data{1,pack_cnt}.train_position_offset_spike_cell = train_position_offset_spike_cell;
                train_data{1,pack_cnt}.train_position_offset = train_position_offset;
                train_data{1,pack_cnt}.train_position_offset_cumsumed_spike_cell = train_position_offset_cumsumed_spike_cell;
                train_data{1,pack_cnt}.train_position_offset_cumsumed_cell = train_position_offset_cumsumed_cell;
                train_data{1,pack_cnt}.train_position_offset_cumsumed = train_position_offset_cumsumed;
                train_data{1,pack_cnt}.train_stroke_spike_cell = train_stroke_spike_cell;
                train_data{1,pack_cnt}.train_stroke_cell = train_stroke_cell;
                train_data{1,pack_cnt}.train_state_unified_stroke_break = train_state_unified_stroke_break;
                train_data{1,pack_cnt}.train_state_separate_stroke_break = train_state_separate_stroke_break;
                
                if if_anaylse_stroke
                    target_stroke_list = trials_data.params.unified_stroke_list;
                    train_target_stroke_data = get_target_stroke_data(target_stroke_list,trial_data(train_trial_index),trials_data.params);
                    train_data{1,pack_cnt}.train_target_stroke_data = train_target_stroke_data;
                end
    
                %%% make test data
                test_spike = [];
                test_PVA = [];
                if if_has_inter_trial
                    test_spike_inter_trial = [];
                    test_PVA_inter_trial = [];
                    test_go_mask = [];
                end
                test_stroke_PVA = [];
                test_stroke_spike = [];
                test_break_PVA = [];
                test_break_spike = [];
                test_state = [];
                test_state_unified_stroke_break = [];
                test_state_separate_stroke_break = [];
                test_absolute_position_spike_cell = [];
                test_absolute_position = [];
                test_position_offset_spike_cell = [];
                test_position_offset = [];
                test_position_offset_cumsumed_spike_cell = [];
                test_position_offset_cumsumed_cell = [];
                test_position_offset_cumsumed = [];
                test_stroke_spike_cell = [];
                test_stroke_cell = [];
                for test_trial_no = 1:numel(test_trial_index)
                    tmp_trial_index = test_trial_index(test_trial_no);
                    test_spike = [test_spike trial_data{1,tmp_trial_index}.spike_bin];
                    test_PVA = [test_PVA trial_data{1,tmp_trial_index}.PVA_bin];
                    if if_has_inter_trial
                        test_spike_inter_trial = [test_spike_inter_trial trial_data{1,tmp_trial_index}.spike_bin_inter_trial];
                        test_PVA_inter_trial = [test_PVA_inter_trial trial_data{1,tmp_trial_index}.PVA_bin_inter_trial];
                        test_go_mask = [test_go_mask trial_data{1,tmp_trial_index}.go_mask];
                    end
                    test_stroke_PVA = [test_stroke_PVA trial_data{1,tmp_trial_index}.stroke_PVA];
                    test_stroke_spike = [test_stroke_spike trial_data{1,tmp_trial_index}.stroke_spike];
                    test_break_PVA = [test_break_PVA trial_data{1,tmp_trial_index}.break_PVA];
                    test_break_spike = [test_break_spike trial_data{1,tmp_trial_index}.break_spike];
                    
                    
                    test_absolute_position_spike_cell{test_trial_no} = trial_data{1,tmp_trial_index}.absolute_position_spike_cell;
                    test_absolute_position{test_trial_no} = trial_data{1,tmp_trial_index}.absolute_position;
                    
                    test_position_offset_spike_cell{test_trial_no} = trial_data{1,tmp_trial_index}.position_offset_spike_cell;
                    test_position_offset{test_trial_no} = trial_data{1,tmp_trial_index}.position_offset;
                    test_position_offset_cumsumed_spike_cell{test_trial_no} = trial_data{1,tmp_trial_index}.position_offset_cumsumed_spike_cell;
                    test_position_offset_cumsumed_cell{test_trial_no} = trial_data{1,tmp_trial_index}.position_offset_cumsumed_cell;
                    test_position_offset_cumsumed{test_trial_no} = trial_data{1,tmp_trial_index}.position_offset_cumsumed;
    
                    test_stroke_spike_cell{test_trial_no} = trial_data{1,tmp_trial_index}.stroke_spike_cell;
                    test_stroke_cell{test_trial_no} = trial_data{1,tmp_trial_index}.stroke_cell;
                    
                    test_state = [test_state trial_data{1,tmp_trial_index}.state_bin];
                    if trials_data.params.if_analyse_stroke
                        test_state_unified_stroke_break = [test_state_unified_stroke_break trial_data{1,tmp_trial_index}.state_unified_stroke_break];
                        test_state_separate_stroke_break = [test_state_separate_stroke_break trial_data{1,tmp_trial_index}.state_separate_stroke_break];
                    end
                end
                
                
                test_data{1,pack_cnt}.test_spike = test_spike;
                test_data{1,pack_cnt}.test_PVA = test_PVA;
                if if_has_inter_trial
                    test_data{1,pack_cnt}.test_spike_inter_trial = test_spike_inter_trial;
                    test_data{1,pack_cnt}.test_PVA_inter_trial = test_PVA_inter_trial;
                    test_data{1,pack_cnt}.test_go_mask = test_go_mask;
                end
                test_data{1,pack_cnt}.test_stroke_PVA = test_stroke_PVA;
                test_data{1,pack_cnt}.test_stroke_spike = test_stroke_spike;
                test_data{1,pack_cnt}.test_break_PVA = test_break_PVA;
                test_data{1,pack_cnt}.test_break_spike = test_break_spike;
                test_data{1,pack_cnt}.test_state = test_state;
                test_data{1,pack_cnt}.test_state_unified_stroke_break = test_state_unified_stroke_break;
                test_data{1,pack_cnt}.test_state_separate_stroke_break = test_state_separate_stroke_break;
                test_data{1,pack_cnt}.test_absolute_position_spike_cell = test_absolute_position_spike_cell;
                test_data{1,pack_cnt}.test_absolute_position = test_absolute_position;
                test_data{1,pack_cnt}.test_position_offset_spike_cell = test_position_offset_spike_cell;
                test_data{1,pack_cnt}.test_position_offset = test_position_offset;
                test_data{1,pack_cnt}.test_position_offset_cumsumed_spike_cell = test_position_offset_cumsumed_spike_cell;
                test_data{1,pack_cnt}.test_position_offset_cumsumed_cell = test_position_offset_cumsumed_cell;
                test_data{1,pack_cnt}.test_position_offset_cumsumed = test_position_offset_cumsumed;
                test_data{1,pack_cnt}.test_stroke_spike_cell = test_stroke_spike_cell;
                test_data{1,pack_cnt}.test_stroke_cell = test_stroke_cell;
                
                if if_anaylse_stroke
                    target_stroke_list = trials_data.params.unified_stroke_list;
                    test_target_stroke_data = get_target_stroke_data(target_stroke_list,trial_data(test_trial_index),trials_data.params);
                    test_data{1,pack_cnt}.test_target_stroke_data = test_target_stroke_data;
                end
    
    
    
                if if_has_val
                    %%% make val data
                    val_spike = [];
                    val_PVA = [];
                    if if_has_inter_trial
                        val_spike_inter_trial = [];
                        val_PVA_inter_trial = [];
                        val_go_mask = [];
                    end
                    val_stroke_PVA = [];
                    val_stroke_spike = [];
                    val_break_PVA = [];
                    val_break_spike = [];
                    val_state = [];
                    val_state_unified_stroke_break = [];
                    val_state_separate_stroke_break = [];
                    val_absolute_position_spike_cell = [];
                    val_absolute_position = [];
                    val_position_offset_spike_cell = [];
                    val_position_offset = [];
                    val_position_offset_cumsumed_spike_cell = [];
                    val_position_offset_cumsumed_cell = [];
                    val_position_offset_cumsumed = [];
                    val_stroke_spike_cell = [];
                    val_stroke_cell = [];
                    for val_trial_no = 1:numel(val_trial_index)
                        tmp_trial_index = val_trial_index(val_trial_no);
                        val_spike = [val_spike trial_data{1,tmp_trial_index}.spike_bin];
                        val_PVA = [val_PVA trial_data{1,tmp_trial_index}.PVA_bin];
                        if if_has_inter_trial
                            val_spike_inter_trial = [val_spike_inter_trial trial_data{1,tmp_trial_index}.spike_bin_inter_trial];
                            val_PVA_inter_trial = [val_PVA_inter_trial trial_data{1,tmp_trial_index}.PVA_bin_inter_trial];
                            val_go_mask = [val_go_mask trial_data{1,tmp_trial_index}.go_mask];
                        end
                        val_stroke_PVA = [val_stroke_PVA trial_data{1,tmp_trial_index}.stroke_PVA];
                        val_stroke_spike = [val_stroke_spike trial_data{1,tmp_trial_index}.stroke_spike];
                        val_break_PVA = [val_break_PVA trial_data{1,tmp_trial_index}.break_PVA];
                        val_break_spike = [val_break_spike trial_data{1,tmp_trial_index}.break_spike];
    
                        val_absolute_position_spike_cell{val_trial_no} = trial_data{1,tmp_trial_index}.absolute_position_spike_cell;
                        val_absolute_position{val_trial_no} = trial_data{1,tmp_trial_index}.absolute_position;
    
                        val_position_offset_spike_cell{val_trial_no} = trial_data{1,tmp_trial_index}.position_offset_spike_cell;
                        val_position_offset{val_trial_no} = trial_data{1,tmp_trial_index}.position_offset;
                        val_position_offset_cumsumed_spike_cell{val_trial_no} = trial_data{1,tmp_trial_index}.position_offset_cumsumed_spike_cell;
                        val_position_offset_cumsumed_cell{val_trial_no} = trial_data{1,tmp_trial_index}.position_offset_cumsumed_cell;
                        val_position_offset_cumsumed{val_trial_no} = trial_data{1,tmp_trial_index}.position_offset_cumsumed;
    
                        val_stroke_spike_cell{val_trial_no} = trial_data{1,tmp_trial_index}.stroke_spike_cell;
                        val_stroke_cell{val_trial_no} = trial_data{1,tmp_trial_index}.stroke_cell;
    
                        val_state = [val_state trial_data{1,tmp_trial_index}.state_bin];
                        if trials_data.params.if_analyse_stroke
                            val_state_unified_stroke_break = [val_state_unified_stroke_break trial_data{1,tmp_trial_index}.state_unified_stroke_break];
                            val_state_separate_stroke_break = [val_state_separate_stroke_break trial_data{1,tmp_trial_index}.state_separate_stroke_break];
                        end
                    end
    
                    val_data{1,pack_cnt}.val_spike = val_spike;
                    val_data{1,pack_cnt}.val_PVA = val_PVA;
                    if if_has_inter_trial
                        val_data{1,pack_cnt}.val_spike_inter_trial = val_spike_inter_trial;
                        val_data{1,pack_cnt}.val_PVA_inter_trial = val_PVA_inter_trial;
                        val_data{1,pack_cnt}.val_go_mask = val_go_mask;
                    end
                    val_data{1,pack_cnt}.val_stroke_PVA = val_stroke_PVA;
                    val_data{1,pack_cnt}.val_stroke_spike = val_stroke_spike;
                    val_data{1,pack_cnt}.val_break_PVA = val_break_PVA;
                    val_data{1,pack_cnt}.val_break_spike = val_break_spike;
                    val_data{1,pack_cnt}.val_state = val_state;
    
                    val_data{1,pack_cnt}.val_absolute_position_spike_cell = val_absolute_position_spike_cell;
                    val_data{1,pack_cnt}.val_absolute_position = val_absolute_position;
                    val_data{1,pack_cnt}.val_position_offset_spike_cell = val_position_offset_spike_cell;
                    val_data{1,pack_cnt}.val_position_offset = val_position_offset;
                    val_data{1,pack_cnt}.val_position_offset_cumsumed_spike_cell = val_position_offset_cumsumed_spike_cell;
                    val_data{1,pack_cnt}.val_position_offset_cumsumed_cell = val_position_offset_cumsumed_cell;
                    val_data{1,pack_cnt}.val_position_offset_cumsumed = val_position_offset_cumsumed;
                    val_data{1,pack_cnt}.val_stroke_spike_cell = val_stroke_spike_cell;
                    val_data{1,pack_cnt}.val_stroke_cell = val_stroke_cell;
                    val_data{1,pack_cnt}.val_state_unified_stroke_break = val_state_unified_stroke_break;
                    val_data{1,pack_cnt}.val_state_separate_stroke_break = val_state_separate_stroke_break;
                    if if_anaylse_stroke
                        target_stroke_list = trials_data.params.unified_stroke_list;
                        val_target_stroke_data = get_target_stroke_data(target_stroke_list,trial_data(val_trial_index),trials_data.params);
                        val_data{1,pack_cnt}.val_target_stroke_data = val_target_stroke_data;
                    end
                end
                pack_cnt = pack_cnt + 1;
            end
        end
        train_test_package.train_data = train_data;
        train_test_package.test_data = test_data;
        if if_has_val
            train_test_package.val_data = val_data;
        end
    else
        disp(['Different target repeat num: ' num2str(target_repeat_num)]);
    end
end

