function [train_test_package] = encode_train_test_packages_binary_DyLearn(train_test_package,trials_data,encoder_type,encode_pack_list)
    

    rng(888)
    

    options = get_DyLearn_encoding_options();
    if trials_data.params.if_replace_true_data_with_simulated
        options.err_smooth_winsize = trials_data.params.DyLearn_error_smooth;

        options.err_smooth_winsize = 5;
        options.keep_percent = 0.1;
        options.min_iter_num = 50;


    end

    encode_pack_num = numel(encode_pack_list);
    GP_encoder_loss_times_mean = [];
    for encode_pack_no = 1:encode_pack_num
        select_pack_index = encode_pack_list(encode_pack_no);
        target_stroke_list = trials_data.params.unified_stroke_list;
        target_stroke_list_num = numel(target_stroke_list);

        plot_info.unique_char_list = trials_data.unique_char_list;
        plot_info.trial_target_list = trials_data.trial_target_list;
        plot_info.trial_target_length_list = trials_data.trial_target_length_list;
        plot_info.trial_target_stroke_length_list = trials_data.trial_target_stroke_length_list;
        plot_info.train_trial_index = train_test_package.train_data{1,select_pack_index}.train_trial_index;
        plot_info.train_state = train_test_package.train_data{1,select_pack_index}.train_state;
        plot_info.train_vel = train_test_package.train_data{1,select_pack_index}.train_PVA(3:4,:);
        plot_info.train_neu = train_test_package.train_data{1,select_pack_index}.train_spike;

        
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
        plot_info.neuron_type = -1;
        plot_info.params = trials_data.params;
        options.plot_info = plot_info;


        %% train encoders
        %%% 用DyLearn训练多个模型
        options.encoder_type = encoder_type;
        options.group_no = 1;
        options.group_model_num = length(find(options.encoder_type(options.group_no,:)~=0));

        PVA_train_gt = train_test_package.train_data{1,select_pack_index}.train_stroke_PVA;
        X_train_gt = train_test_package.train_data{1,select_pack_index}.train_stroke_PVA(3:4,:);
        Y_train_gt = train_test_package.train_data{1,select_pack_index}.train_stroke_spike;
        [encoder_DyLearn, DyLearn_param_m] = generate_model(X_train_gt,Y_train_gt,options.group_model_num,options);
        [encoder_DyLearn, DyLearn_param_m] = delete_nan_models(encoder_DyLearn,DyLearn_param_m);

        options.plot_info.err_smooth_winsize = options.err_smooth_winsize;
        best_iter_no = DyLearn_param_m.best_iter_no
        options.plot_info.keep_percent = options.keep_percent;
        options.plot_info.if_decode_stroke = 1;
        encoder_type_list = options.encoder_type(options.group_no,:);
        model_assign_table_time = DyLearn_param_m.model_assign_table_time;
        %% plot config
        if_visualize_model_traj = 1; % 画每个模型选到的轨迹
        if_cal_weights_kin_CC = 0; % 画每个模型权重和pva之间的CC
        if_observe_DyLearn_model_data = 0;

        %% visualize the traj selected by different models 
        % 可视化每个模型选到的数据的轨迹
        if if_visualize_model_traj
            

            if trials_data.params.if_replace_true_data_with_simulated
                train_simulated_context_index = train_test_package.train_data{1,select_pack_index}.train_simulated_context_index;
                train_simulated_context_index_no_break = train_simulated_context_index(logical(train_test_package.train_data{1,select_pack_index}.train_state));

                simulate_context_num = trials_data.params.simulate_context_num;
                simulated_context_assign_table = zeros(simulate_context_num,numel(train_simulated_context_index_no_break));
                for simulate_context_no = 1:simulate_context_num
                    simulated_context_assign_table(simulate_context_no,train_simulated_context_index_no_break==simulate_context_no) = 1;
                end

                %%% 画仿真的dyLearn
                for plot_iter_no = [best_iter_no]
                    min_total_err = DyLearn_param_m.total_err_iter(plot_iter_no);
                    options.plot_info.min_total_err = min_total_err;
                    options.plot_info.plot_iter_no = plot_iter_no;
                    options.plot_info.model_assign_table_GT = 0;
                    options.plot_info.assign_table_GT = simulated_context_assign_table;
    
                    return_info = plot_model_assign_with_traj(X_train_gt,model_assign_table_time{plot_iter_no},encoder_type_list,options.plot_info)
                %     plot_model_assign_with_velocity(trainKinData,model_assign_table_time{plot_iter_no},encoder_type_list,options.plot_info)
        
                %     plot_error_with_traj(trainKinData,model_assign_table_time{plot_iter_no},encoder_type_list,options.plot_info)
                %     plot_error_with_velocity(trainKinData,model_assign_table_time{plot_iter_no},encoder_type_list,options.plot_info)
                
                    train_test_package.model_assign_acc = return_info.model_assign_acc;
                end

                %%% 画仿真的Ground truth
                options.plot_info.model_assign_table_GT = 1;
                plot_model_assign_with_traj(X_train_gt,simulated_context_assign_table,encoder_type_list,options.plot_info)
            else
                for plot_iter_no = [best_iter_no]
                    min_total_err = DyLearn_param_m.total_err_iter(plot_iter_no);
                    options.plot_info.min_total_err = min_total_err;
                    options.plot_info.plot_iter_no = plot_iter_no;
                    options.plot_info.model_assign_table_GT = 0;
    
                    options.plot_info.params.simulate_awgn_snr = 0;
%                     return_info = plot_model_assign_with_traj(X_train_gt,model_assign_table_time{plot_iter_no},encoder_type_list,options.plot_info)

                    return_info = plot_model_assign_with_W(train_test_package,X_train_gt,model_assign_table_time{plot_iter_no},encoder_type_list,options.plot_info)

                %     plot_model_assign_with_velocity(trainKinData,model_assign_table_time{plot_iter_no},encoder_type_list,options.plot_info)
        
                %     plot_error_with_traj(trainKinData,model_assign_table_time{plot_iter_no},encoder_type_list,options.plot_info)
                %     plot_error_with_velocity(trainKinData,model_assign_table_time{plot_iter_no},encoder_type_list,options.plot_info)
                
                end

            end
        end

        %% cal CC between model assignment weights and kinematics
        % 计算模型的权重或者分配0、1与数据pva之间的CC
        if if_cal_weights_kin_CC
            for plot_iter_no = [best_iter_no]
                min_total_err = DyLearn_param_m.total_err_iter(plot_iter_no);
                options.plot_info.min_total_err = min_total_err;
                options.plot_info.plot_iter_no = plot_iter_no;

                plot_weights_kin_CC(PVA_train_gt,model_assign_table_time{plot_iter_no},encoder_type_list,options.plot_info)
         
            end
        end

        %% observe data
        % 
        if if_observe_DyLearn_model_data
            for plot_iter_no = [best_iter_no]
                min_total_err = DyLearn_param_m.total_err_iter(plot_iter_no);
                options.plot_info.min_total_err = min_total_err;
                options.plot_info.plot_iter_no = plot_iter_no;

                observe_DyLearn_model_data(PVA_train_gt,model_assign_table_time{plot_iter_no},encoder_type_list,options.plot_info)
         
            end

        end





        %% visualize model iters




        %% test encoders
        X_test_gt = train_test_package.test_data{1,select_pack_index}.test_stroke_PVA(3:4,:);
        Y_test_gt = train_test_package.test_data{1,select_pack_index}.test_stroke_spike;
        state_test = train_test_package.test_data{1,select_pack_index}.test_state;
        point_num = size(X_test_gt,2);

        Y_predict_DyLearn_all_model = [];
        Y_predict_DyLearn = [];

        %%% encoder_type 3: DyLearn 
        DyLearn_model_num = numel(encoder_DyLearn);
        for DyLearn_model_no = 1:DyLearn_model_num
            tmp_DyLearn_model = encoder_DyLearn{1,DyLearn_model_no};
            tmp_DyLearn_model_type = tmp_DyLearn_model.type;
            switch tmp_DyLearn_model_type
                case 'linear'
                    [y_decoded_hasbias,y_decoded_nobias] = decode_linear(X_test_gt,tmp_DyLearn_model);
                    if options.bias_Q
                        neu_predict_DyLearn_one_model = y_decoded_hasbias;
                    else
                        neu_predict_DyLearn_one_model = y_decoded_nobias;
                    end
                case 'nn'
                    neu_predict_DyLearn_one_model = decode_nn(X_test_gt,tmp_DyLearn_model);
            end
            % 每一行是一个模型 每一列是一个神经元 每一页是一个粒子
            Y_predict_DyLearn_all_model(DyLearn_model_no,:,:) = neu_predict_DyLearn_one_model;
        end
        Y_test_gt_reshape = reshape(Y_test_gt,[1,size(Y_test_gt)]);
        residual_DyLearn_all_model = Y_test_gt_reshape - Y_predict_DyLearn_all_model;

        residual_DyLearn = squeeze(min(abs(residual_DyLearn_all_model),[],1));
        mse_DyLearn = residual_DyLearn .* residual_DyLearn;
        train_test_package.test_data{1,select_pack_index}.encode_mse_DyLearn = mse_DyLearn;
        train_test_package.test_data{1,select_pack_index}.encode_spike_DyLearn = Y_predict_DyLearn;
        train_test_package.test_data{1,select_pack_index}.DyLearn_param_m = DyLearn_param_m;
        
    end
    
    

end

