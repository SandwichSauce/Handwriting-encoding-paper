function decoded_info = decode_stroke_DyLearn(if_select_neurons,select_neuron_list,train_stroke_cell,train_stroke_spike_cell,test_stroke_cell,test_stroke_spike_cell,plot_info)

    options = get_DyLearn_options();
    if plot_info.if_replace_true_data_with_simulated
        options.err_smooth_winsize = plot_info.params.DyLearn_error_smooth;
        options.keep_percent = 0.2;
        options.bias_mse = 1;
        options.bias_Q = 1;
    end
    options.sequence_length = plot_info.sequence_length;
    options.encoder_type = plot_info.encoder_type;
    options.linear_prior_threshold = plot_info.linear_prior_threshold;
    options.group_num = size(options.encoder_type,1);
    options.model_num = numel(options.encoder_type);
    options.short_date_session = plot_info.short_date_session;


    %%% 对于每一个stroke，取cell
    sequence_length = options.sequence_length;
    train_fr_sequence_concat = [];
    train_kin = [];
    cell_cnt = 1;
    train_trial_num = numel(train_stroke_spike_cell);
    for train_trial_no = 1:train_trial_num
        tmp_trial_spike_cell = train_stroke_spike_cell{1,train_trial_no};
        tmp_trial_kin_cell = train_stroke_cell{1,train_trial_no};
        tmp_stroke_num = numel(tmp_trial_spike_cell);
        for tmp_stroke_no = 1:tmp_stroke_num
            tmp_train_fr = tmp_trial_spike_cell{1,tmp_stroke_no};
            if if_select_neurons
                tmp_train_fr = tmp_train_fr(select_neuron_list,:);
            end
            tmp_train_kin = tmp_trial_kin_cell{1,tmp_stroke_no};
            train_kin = [train_kin tmp_train_kin];
            cell_num = size(tmp_train_fr,2);
            for cell_no = 1:cell_num
                tmp_mask = max(1,(cell_no-sequence_length+1)):cell_no;
                tmp_fr = tmp_train_fr(:,tmp_mask);
                tmp_fr_point_num = size(tmp_fr,2);
                if tmp_fr_point_num < sequence_length
                    tmp_fr = [zeros(size(tmp_fr,1),(sequence_length-tmp_fr_point_num)) tmp_fr];
                end
                train_fr_sequence_concat = [train_fr_sequence_concat reshape(tmp_fr,[],1)];
                cell_cnt = cell_cnt + 1;
            end
        end
    end

    
    test_fr_sequence_concat = [];
    test_kin = [];
    cell_cnt = 1;
    test_trial_num = numel(test_stroke_spike_cell);
    for test_trial_no = 1:test_trial_num
        tmp_trial_spike_cell = test_stroke_spike_cell{1,test_trial_no};
        tmp_trial_kin_cell = test_stroke_cell{1,test_trial_no};
        tmp_stroke_num = numel(tmp_trial_spike_cell);
        for tmp_stroke_no = 1:tmp_stroke_num
            tmp_test_fr = tmp_trial_spike_cell{1,tmp_stroke_no};
            if if_select_neurons
                tmp_test_fr = tmp_test_fr(select_neuron_list,:);
            end
            tmp_test_kin = tmp_trial_kin_cell{1,tmp_stroke_no};
            test_kin = [test_kin tmp_test_kin];
            cell_num = size(tmp_test_fr,2);
            for cell_no = 1:cell_num
                tmp_mask = max(1,(cell_no-sequence_length+1)):cell_no;
                tmp_fr = tmp_test_fr(:,tmp_mask);
                tmp_fr_point_num = size(tmp_fr,2);
                if tmp_fr_point_num < sequence_length
                    tmp_fr = [zeros(size(tmp_fr,1),(sequence_length-tmp_fr_point_num)) tmp_fr];
                end
                test_fr_sequence_concat = [test_fr_sequence_concat reshape(tmp_fr,[],1)];
                cell_cnt = cell_cnt + 1;
            end
        end
    end

    X_train = train_kin;
    Y_train = train_fr_sequence_concat;
    X_test = test_kin;
    Y_test = test_fr_sequence_concat;

    decoded_info = decode_pack_DyLearn(Y_train, X_train, Y_test, X_test, plot_info);

    predicted_stroke_res = cal_RMSE_CC_VAF_2D(X_test(1:2,:),decoded_info.X_predict(1:2,:));
    decoded_info.predicted_stroke_res = predicted_stroke_res;

    
end

