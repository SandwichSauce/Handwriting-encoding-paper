function [train_test_package] = decode_train_test_packages_binary(train_test_package,trials_data,decoder_type,decode_pack_list)


    kin_type = trials_data.params.kin_type;
    if_decode_stroke = trials_data.params.if_decode_stroke;
    if_decode_break = trials_data.params.if_decode_break;
    if_decode_position_offset = trials_data.params.if_decode_position_offset;
    if_decode_absolute_position = trials_data.params.if_decode_absolute_position;
    if_select_neurons = trials_data.params.if_select_neurons || trials_data.params.if_analyse_neuron_types;
    if_decode_position_offset_by_cumsum = trials_data.params.if_decode_position_offset_by_cumsum;
    if_sequence_to_one = trials_data.params.if_sequence_to_one;

    if trials_data.params.if_select_neurons 
        select_neuron_list = trials_data.good_neuron_by_svm_list;
    end
    if trials_data.params.if_analyse_neuron_types
        select_neuron_list = trials_data.neuron_type_list;
    end
    
    X_predict_all = [];
    X_test_all = [];
    abs_pos_test_all = [];
    abs_pos_predict_all = [];
    offset_pos_test_all = [];
    offset_pos_predict_all = [];
    offset_pos_cumsumed_test_all = [];
    offset_pos_cumsumed_predict_all = [];

    
    decode_pack_num = numel(decode_pack_list);
    for decode_pack_no = 1:decode_pack_num
        select_pack_index = decode_pack_list(decode_pack_no);
        plot_info.unique_char_list = trials_data.unique_char_list;
        plot_info.trial_target_list = trials_data.trial_target_list;
        plot_info.trial_target_length_list = trials_data.trial_target_length_list;
        plot_info.trial_target_stroke_length_list = trials_data.trial_target_stroke_length_list;
        plot_info.train_trial_index = train_test_package.train_data{1,select_pack_index}.train_trial_index;
        plot_info.train_state = train_test_package.train_data{1,select_pack_index}.train_state;
        plot_info.train_vel = train_test_package.train_data{1,select_pack_index}.train_PVA(3:4,:);
        plot_info.kin_type = trials_data.params.kin_type;
        plot_info.traj_type = trials_data.params.traj_type;
        plot_info.if_in_E_disk = trials_data.params.if_in_E_disk;
        plot_info.if_label_type_is_cell = trials_data.params.if_label_type_is_cell;
        plot_info.short_date_session = trials_data.params.short_date_session;
        plot_info.save_path = trials_data.params.save_path;
        plot_info.sequence_length = trials_data.params.sequence_length;
        plot_info.encoder_type = trials_data.params.encoder_type;
        plot_info.linear_prior_threshold = trials_data.params.linear_prior_threshold;
        plot_info.short_date_session = trials_data.params.short_date_session;
        plot_info.select_pack_index = select_pack_index;
        plot_info.if_replace_true_data_with_simulated = trials_data.params.if_replace_true_data_with_simulated;
        plot_info.params = trials_data.params;
                
        if if_decode_stroke == 0 && if_decode_break == 0 % 书写断笔合在一起解
            X_train = train_test_package.train_data{1,select_pack_index}.train_PVA;
            Y_train = train_test_package.train_data{1,select_pack_index}.train_spike;
            X_test = train_test_package.test_data{1,select_pack_index}.test_PVA;
            Y_test = train_test_package.test_data{1,select_pack_index}.test_spike;
        else % 解书写
            X_train = train_test_package.train_data{1,select_pack_index}.train_stroke_PVA;
            Y_train = train_test_package.train_data{1,select_pack_index}.train_stroke_spike;
            X_test = train_test_package.test_data{1,select_pack_index}.test_stroke_PVA;
            Y_test = train_test_package.test_data{1,select_pack_index}.test_stroke_spike;
        end
        
        if strcmp(kin_type,'PVA')
            X_train = X_train;
            X_test = X_test;
        elseif strcmp(kin_type,'P')
            X_train = X_train(1:2,:);
            X_test = X_test(1:2,:);
        elseif strcmp(kin_type,'V')
            X_train = X_train(3:4,:);
            X_test = X_test(3:4,:);
        elseif strcmp(kin_type,'VS')
            X_train = [X_train(3:4,:);state_separate_stroke_break_train];
            X_test = [X_test(3:4,:);state_separate_stroke_break_test];
        end

        if if_select_neurons
            Y_train = Y_train(select_neuron_list,:);
            Y_test = Y_test(select_neuron_list,:);
        else
            select_neuron_list = [];
        end
        
        if if_decode_stroke == 1
            train_stroke_cell = train_test_package.train_data{1,select_pack_index}.train_stroke_cell;            
            train_stroke_spike_cell = train_test_package.train_data{1,select_pack_index}.train_stroke_spike_cell;
            test_stroke_cell = train_test_package.test_data{1,select_pack_index}.test_stroke_cell;            
            test_stroke_spike_cell = train_test_package.test_data{1,select_pack_index}.test_stroke_spike_cell;
            if strcmp(decoder_type,'LSTM')
                if if_sequence_to_one
                    decoded_info = decode_stroke_LSTM(if_select_neurons,select_neuron_list,train_stroke_cell,train_stroke_spike_cell,test_stroke_cell,test_stroke_spike_cell,plot_info);
                end
            elseif strcmp(decoder_type,'KF')
                if if_sequence_to_one
                    decoded_info = decode_stroke_KF(if_select_neurons,select_neuron_list,train_stroke_cell,train_stroke_spike_cell,test_stroke_cell,test_stroke_spike_cell,plot_info);
                else
                    decoded_info = decode_pack_KF(Y_train, X_train, Y_test, X_test);
                end
            elseif strcmp(decoder_type,'TCN')
                if if_sequence_to_one
                    decoded_info = decode_pack_TCN(Y_train, X_train, Y_test, X_test);
                end
            elseif strcmp(decoder_type,'DyLearn')
                if if_sequence_to_one
                    decoded_info = decode_stroke_DyLearn(if_select_neurons,select_neuron_list,train_stroke_cell,train_stroke_spike_cell,test_stroke_cell,test_stroke_spike_cell,plot_info);
                else
                    decoded_info = decode_pack_DyLearn(Y_train, X_train, Y_test, X_test, plot_info);
                end
            end
        end
        if if_decode_break == 1 % 单独解断笔
            if if_decode_absolute_position
                %%% decode absolute position
                train_absolute_position = train_test_package.train_data{1,select_pack_index}.train_absolute_position;
                train_absolute_position_spike_cell = train_test_package.train_data{1,select_pack_index}.train_absolute_position_spike_cell;
                test_absolute_position = train_test_package.test_data{1,select_pack_index}.test_absolute_position;
                test_absolute_position_spike_cell = train_test_package.test_data{1,select_pack_index}.test_absolute_position_spike_cell;
                if strcmp(decoder_type,'LSTM')
                    [predicted_absolute_position,predicted_absolute_position_res,abs_pos_test,abs_pos_predict] = decode_absolute_position_LSTM(train_absolute_position,train_absolute_position_spike_cell,test_absolute_position,test_absolute_position_spike_cell);
                end
            end
            if if_decode_position_offset
                %%% decode position offset
                train_position_offset = train_test_package.train_data{1,select_pack_index}.train_position_offset;            
                train_position_offset_spike_cell = train_test_package.train_data{1,select_pack_index}.train_position_offset_spike_cell;
                test_position_offset = train_test_package.test_data{1,select_pack_index}.test_position_offset;            
                test_position_offset_spike_cell = train_test_package.test_data{1,select_pack_index}.test_position_offset_spike_cell;
                if strcmp(decoder_type,'LSTM')
                    [predicted_position_offset,predicted_position_offset_res,offset_pos_test,offset_pos_predict] = decode_position_offset_LSTM(train_position_offset,train_position_offset_spike_cell,test_position_offset,test_position_offset_spike_cell);
                end
                if if_decode_absolute_position == 0
                    predicted_absolute_position = predicted_position_offset;
                    predicted_absolute_position_res = 0;
                    abs_pos_test = [0;0];
                    abs_pos_predict = [0;0];
                end
            end
            if if_decode_position_offset_by_cumsum
                %%% 用不偷懒的方式，把每个断笔先取cell再拼起来
                %%% decode position offset by cumsum
                train_position_offset_cumsumed_cell = train_test_package.train_data{1,select_pack_index}.train_position_offset_cumsumed_cell;            
                train_position_offset_cumsumed_spike_cell = train_test_package.train_data{1,select_pack_index}.train_position_offset_cumsumed_spike_cell;
                test_position_offset_cumsumed_cell = train_test_package.test_data{1,select_pack_index}.test_position_offset_cumsumed_cell;            
                test_position_offset_cumsumed_spike_cell = train_test_package.test_data{1,select_pack_index}.test_position_offset_cumsumed_spike_cell;
                if strcmp(decoder_type,'LSTM')
                    if if_sequence_to_one
                        decoded_info_position_offset_cumsumed = decode_position_offset_cumsumed_LSTM(if_select_neurons,select_neuron_list,train_position_offset_cumsumed_cell,train_position_offset_cumsumed_spike_cell,test_position_offset_cumsumed_cell,test_position_offset_cumsumed_spike_cell);
                    end
                elseif strcmp(decoder_type,'KF')
                    if if_sequence_to_one
                        decoded_info_position_offset_cumsumed = decode_position_offset_cumsumed_KF(if_select_neurons,select_neuron_list,train_position_offset_cumsumed_cell,train_position_offset_cumsumed_spike_cell,test_position_offset_cumsumed_cell,test_position_offset_cumsumed_spike_cell);
                    end
                elseif strcmp(decoder_type,'DyLearn')
                    if if_sequence_to_one
                        decoded_info_position_offset_cumsumed = decode_position_offset_cumsumed_DyLearn(if_select_neurons,select_neuron_list,train_position_offset_cumsumed_cell,train_position_offset_cumsumed_spike_cell,test_position_offset_cumsumed_cell,test_position_offset_cumsumed_spike_cell,plot_info);
                    end
                end
                

                if if_decode_absolute_position == 0
                    predicted_absolute_position = 0;
                    predicted_absolute_position_res = 0;
                    abs_pos_test = [0;0];
                    abs_pos_predict = [0;0];
                end
                if if_decode_position_offset == 0
                    predicted_position_offset = 0;
                    predicted_position_offset_res = 0;
                    offset_pos_test = [0;0];
                    offset_pos_predict = [0;0];
                end
            end
        else
            offset_pos_test = [0;0];
            offset_pos_predict = [0;0];
            abs_pos_test = [0;0];
            abs_pos_predict = [0;0];
            predicted_position_offset_res = 0;
            predicted_absolute_position_res = 0;
        end
        

        if strcmp(kin_type,'VS')
            state_separate_stroke_break_train = train_test_package.train_data{1,select_pack_index}.train_state_separate_stroke_break;
            state_separate_stroke_break_test = train_test_package.test_data{1,select_pack_index}.test_state_separate_stroke_break;
        end
        
        
        X_test_all = [X_test_all X_test];
        abs_pos_test_all = [abs_pos_test_all abs_pos_test];
        abs_pos_predict_all = [abs_pos_predict_all abs_pos_predict];
        offset_pos_test_all = [offset_pos_test_all offset_pos_test];
        offset_pos_predict_all = [offset_pos_predict_all offset_pos_predict];
        if if_decode_position_offset_by_cumsum
            offset_pos_cumsumed_test_all = [offset_pos_cumsumed_test_all decoded_info_position_offset_cumsumed.offset_pos_cumsumed_test];
            offset_pos_cumsumed_predict_all = [offset_pos_cumsumed_predict_all decoded_info_position_offset_cumsumed.offset_pos_cumsumed_predict];
        end

        if if_decode_stroke == 0 && if_decode_break == 1 % 只解断笔
            decoded_info.X_predict = X_test;
            X_predict_all = [X_predict_all decoded_info.X_predict];
        else
            X_predict_all = [X_predict_all decoded_info.X_predict];
        end
        
        if size(X_test,1) == 2 || size(X_test,1) == 3
            mean_rmse_cc_vaf = cal_RMSE_CC_VAF_2D(X_test(1:2,:),decoded_info.X_predict(1:2,:));
        elseif size(X_test,1) == 6 || size(X_test,1) == 7
            evaluation_res = cal_RMSE_CC_VAF_6D(X_test(1:6,:),decoded_info.X_predict(1:6,:));
            mean_rmse_cc_vaf = evaluation_res.vel_mean_rmse_cc_vaf;
            train_test_package.test_data{1,select_pack_index}.evaluation_res = evaluation_res;
        end
        
        pack_VAF = mean_rmse_cc_vaf(end);
        pack_VAF = roundn(pack_VAF,-4);
        pack_VAF_abs = predicted_absolute_position_res(end);
        pack_VAF_offset = predicted_position_offset_res(end);
        
        pack_VAF_abs = roundn(pack_VAF_abs,-4);
        pack_VAF_offset = roundn(pack_VAF_offset,-4);
        train_test_package.test_data{1,select_pack_index}.pack_VAF = pack_VAF;
        train_test_package.test_data{1,select_pack_index}.pack_VAF_abs = pack_VAF_abs;
        train_test_package.test_data{1,select_pack_index}.pack_VAF_offset = pack_VAF_offset;
        train_test_package.test_data{1,select_pack_index}.mean_rmse_cc_vaf = mean_rmse_cc_vaf;
        train_test_package.test_data{1,select_pack_index}.mean_rmse_cc_vaf_abs = predicted_absolute_position_res;
        train_test_package.test_data{1,select_pack_index}.mean_rmse_cc_vaf_offset = predicted_position_offset_res;
        
        train_test_package.test_data{1,select_pack_index}.decoded_info = decoded_info;
        if if_decode_stroke
            if strcmp(decoder_type,'LSTM')
                train_test_package.test_data{1,select_pack_index}.predicted_stroke_cell = decoded_info.predicted_stroke_cell;
            end
        end
        if if_decode_break
            train_test_package.test_data{1,select_pack_index}.predicted_absolute_position = predicted_absolute_position;
            train_test_package.test_data{1,select_pack_index}.predicted_position_offset = predicted_position_offset;
            if if_decode_position_offset_by_cumsum
                train_test_package.test_data{1,select_pack_index}.decoded_info_position_offset_cumsumed = decoded_info_position_offset_cumsumed;
                train_test_package.test_data{1,select_pack_index}.test_position_offset_cumsumed = decoded_info_position_offset_cumsumed.test_position_offset_cumsumed;
                train_test_package.test_data{1,select_pack_index}.predicted_position_offset_cumsumed_cell = decoded_info_position_offset_cumsumed.predicted_position_offset_cumsumed_cell;
                train_test_package.test_data{1,select_pack_index}.predicted_position_offset_cumsumed = decoded_info_position_offset_cumsumed.predicted_position_offset_cumsumed;

                pack_VAF_offset_cumsumed = decoded_info_position_offset_cumsumed.predicted_position_offset_cumsumed_res(end);
                pack_VAF_offset_cumsumed = roundn(pack_VAF_offset_cumsumed,-4);
                train_test_package.test_data{1,select_pack_index}.pack_VAF_offset_cumsumed = pack_VAF_offset_cumsumed;
                train_test_package.test_data{1,select_pack_index}.mean_rmse_cc_vaf_offset_cumsumed = decoded_info_position_offset_cumsumed.predicted_position_offset_cumsumed_res;
%                 train_test_package.test_data{1,select_pack_index}.X_predict_break = X_predict_break;
            end
        end
        
        
    end
    
    if size(X_test_all,1) == 2 || size(X_test_all,1) == 3
        mean_rmse_cc_vaf_all = cal_RMSE_CC_VAF_2D(X_test_all(1:2,:),X_predict_all(1:2,:));
    elseif size(X_test_all,1) == 6 || size(X_test_all,1) == 7
        evaluation_res = cal_RMSE_CC_VAF_6D(X_test_all(1:6,:),X_predict_all(1:6,:));
        mean_rmse_cc_vaf_all = evaluation_res.vel_mean_rmse_cc_vaf;
        train_test_package.evaluation_res = evaluation_res;
    end
    if if_decode_position_offset_by_cumsum
        mean_rmse_cc_vaf_all_offset_cumsumed = cal_RMSE_CC_VAF_2D(offset_pos_cumsumed_test_all(1:2,:),offset_pos_cumsumed_predict_all(1:2,:));
        package_VAF_offset_cumsumed = mean_rmse_cc_vaf_all_offset_cumsumed(end);
        package_VAF_offset_cumsumed = roundn(package_VAF_offset_cumsumed,-4);
        train_test_package.package_VAF_offset_cumsumed = package_VAF_offset_cumsumed;
        train_test_package.mean_rmse_cc_vaf_all_offset_cumsumed = mean_rmse_cc_vaf_all_offset_cumsumed;
    end
    mean_rmse_cc_vaf_all_abs = cal_RMSE_CC_VAF_2D(abs_pos_test_all(1:2,:),abs_pos_predict_all(1:2,:));
    mean_rmse_cc_vaf_all_offset = cal_RMSE_CC_VAF_2D(offset_pos_test_all(1:2,:),offset_pos_predict_all(1:2,:));
    package_VAF_abs = mean_rmse_cc_vaf_all_abs(end);
    package_VAF_abs = roundn(package_VAF_abs,-4);
    package_VAF_offset = mean_rmse_cc_vaf_all_offset(end);
    package_VAF_offset = roundn(package_VAF_offset,-4);
    package_VAF = mean_rmse_cc_vaf_all(end);
    package_VAF = roundn(package_VAF,-4);
    train_test_package.package_VAF = package_VAF;
    train_test_package.package_VAF_abs = package_VAF_abs;
    train_test_package.package_VAF_offset = package_VAF_offset;
    train_test_package.mean_rmse_cc_vaf_all = mean_rmse_cc_vaf_all;
    train_test_package.mean_rmse_cc_vaf_all_abs = mean_rmse_cc_vaf_all_abs;
    train_test_package.mean_rmse_cc_vaf_all_offset = mean_rmse_cc_vaf_all_offset;
    
%     train_test_package.X_test_all= X_test_all;
%     train_test_package.X_predict_all = X_predict_all;

end

