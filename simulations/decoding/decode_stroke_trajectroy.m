clc;clear;close all;
base_path = 'D:\OneDrive - zju.edu.cn\ZXY\handwriting\handwriting encoding paper\';
load([base_path 'simulations\simulated data\2022-06-10-S1-simulated-snr20.mat']);
params = trials_data.params;

%% decoding config 遍历所有的解码模型和不同线性非线性模型数的配置
sequence_length_list = [1];
sequence_length_list_num = numel(sequence_length_list);
if_linear_nn_in_one_group = 0;
linear_prior_threshold_list = [0];
linear_prior_threshold_list_num = numel(linear_prior_threshold_list);
for linear_prior_threshold_list_no = 1:linear_prior_threshold_list_num
   linear_prior_threshold = linear_prior_threshold_list(linear_prior_threshold_list_no)
   params.if_linear_nn_in_one_group = if_linear_nn_in_one_group;
   params.linear_prior_threshold = linear_prior_threshold;
   for sequence_length_list_no = 1:sequence_length_list_num

       sequence_length = sequence_length_list(sequence_length_list_no)
       decoder_linear_nonlinear_list = [2,params.simulate_context_num,0];
    
       decoder_linear_nonlinear_list_num = size(decoder_linear_nonlinear_list,1);
       for decoder_linear_nonlinear_list_no = 1:decoder_linear_nonlinear_list_num
            tmp_decoder_linear_nonlinear = decoder_linear_nonlinear_list(decoder_linear_nonlinear_list_no,:)
            decoder_type_label = tmp_decoder_linear_nonlinear(1);
            if decoder_type_label == 1
                decoder_type = 'KF'; % decoder 1
            elseif decoder_type_label == 2
                decoder_type = 'DyLearn'; % decoder 2
            elseif decoder_type_label == 3
                decoder_type = 'LSTM'; % decoder 2
            end
            linear_model_num = tmp_decoder_linear_nonlinear(2);
            nonlinear_model_num = tmp_decoder_linear_nonlinear(3);
            linear_model_label = ones(1,linear_model_num);
            nonlinear_model_label = ones(1,nonlinear_model_num)*3;
            if if_linear_nn_in_one_group
                encoder_type = [linear_model_label nonlinear_model_label];
            else
                max_model_num = max(linear_model_num,nonlinear_model_num);
                linear_model_label = [linear_model_label zeros(1,max_model_num-linear_model_num)];
                nonlinear_model_label = [nonlinear_model_label zeros(1,max_model_num-nonlinear_model_num)];
                if linear_model_num == 0
                    linear_model_label = [];
                end
                if nonlinear_model_num == 0
                    nonlinear_model_label = [];
                end
                encoder_type = [linear_model_label;nonlinear_model_label];
            end
            params.linear_model_num = linear_model_num;
            params.nonlinear_model_num = nonlinear_model_num;
            params.encoder_type = encoder_type;
            params.sequence_length = sequence_length;
           %% divide data: train val test 
           % divide_type = 'leave-one-trial-out';
            divide_type = 'leave-one-repeat-out';
            % divide_type = 'leave-one-char-out';
            params.divide_type = divide_type;
            
            %%% make train test data
            train_test_package = divide_train_test_packages(trials_data,divide_type);%get train and test data
        
            %% decoding config
            decode_pack_list = 1:numel(train_test_package.train_data);
%                 decode_pack_list = 1; 
            
            % arrange_type = 'unsort';
            arrange_type = 'sort';    
            %% 获取当前时间戳，为了保存pack和package和图片的一致性
            cur_time_stamp = datestr(now,'MMSS')
            params.cur_time_stamp = cur_time_stamp;
        
            
            %% 保存路径
            if params.if_cue
                cue_type = 'cue';
            else
                cue_type = 'no cue';
            end
        
            if params.if_offline
                sort_type = 'offline_sorted';
            else
                sort_type = 'online_sorted';
            end
        
            save_path = [base_path 'workspace\' cue_type '\' sort_type '\' divide_type '\encoding_paper\simulation\' params.short_date_session '\' decoder_type '\'];
          
            if exist(save_path,'dir')==0
                mkdir(save_path);
            end
            
            params.save_path = save_path;
            trials_data.params = params;
        
            %% 对所有{训练，测试}做解码
            params.decoder_type = decoder_type;
            params.decode_pack_list = decode_pack_list;
            
            if params.if_use_bi_decoder
                [train_test_package] = decode_train_test_packages_binary(train_test_package,trials_data,decoder_type,decode_pack_list);
            end
        
            train_test_package.params = params;
            
            %% plot traj
        %     plot_train_test_packages_stroke_only(train_test_package,arrange_type);
            decoded_data = plot_stroke_only_get_save_info(train_test_package,arrange_type);
                
            %% 保存workspace
            decode_pack_num = numel(decode_pack_list);
            for decode_pack_no = 1:decode_pack_num
                train_test_pack = [];
                select_pack_index = decode_pack_list(decode_pack_no);
    
                decoded_data.fold_info{1,select_pack_index}.train_trial_index = train_test_package.train_data{1,select_pack_index}.train_trial_index;
                decoded_data.fold_info{1,select_pack_index}.test_trial_index = train_test_package.test_data{1,select_pack_index}.test_trial_index;
                decoded_data.fold_info{1,select_pack_index}.mean_rmse_cc_vaf = train_test_package.test_data{1,select_pack_index}.mean_rmse_cc_vaf;
    
                pack_VAF = train_test_package.test_data{1,select_pack_index}.pack_VAF;
                pack_VAF_abs = train_test_package.test_data{1,select_pack_index}.pack_VAF_abs;
                pack_VAF_offset = train_test_package.test_data{1,select_pack_index}.pack_VAF_offset;
                if params.if_decode_position_offset_by_cumsum
                    pack_VAF_offset_cumsumed = train_test_package.test_data{1,select_pack_index}.pack_VAF_offset_cumsumed;
                else
                    pack_VAF_offset_cumsumed = 0;
                end
                if pack_VAF_offset == 0 && params.if_decode_position_offset_by_cumsum
                    pack_VAF_offset = pack_VAF_offset_cumsumed;
                end
                train_test_pack.train_data = train_test_package.train_data{1,select_pack_index};
                train_test_pack.test_data = train_test_package.test_data{1,select_pack_index};
                save_name_pack = [save_path 'linear' num2str(linear_model_num) '-nn' num2str(nonlinear_model_num) '-sequence' num2str(sequence_length) '-pack' num2str(select_pack_index) '.mat'];
    
                save(save_name_pack, 'train_test_pack');
            end
            
            package_VAF = train_test_package.package_VAF;
            package_VAF_abs = train_test_package.package_VAF_abs;
            package_VAF_offset = train_test_package.package_VAF_offset; 
            decoded_data.mean_rmse_cc_vaf_all = train_test_package.mean_rmse_cc_vaf_all;
            if params.if_decode_position_offset_by_cumsum
                package_VAF_offset_cumsumed = train_test_package.package_VAF_offset_cumsumed;
            else
                package_VAF_offset_cumsumed = 0;
            end
            if package_VAF_offset == 0 && params.if_decode_position_offset_by_cumsum
                package_VAF_offset = package_VAF_offset_cumsumed;
            end
            mean_rmse_cc_vaf_all = train_test_package.mean_rmse_cc_vaf_all;
            mean_rmse_cc_vaf_all_abs = train_test_package.mean_rmse_cc_vaf_all_abs;
            mean_rmse_cc_vaf_all_offset = train_test_package.mean_rmse_cc_vaf_all_offset;
            if params.if_decode_position_offset_by_cumsum
                mean_rmse_cc_vaf_all_offset_cumsumed = train_test_package.mean_rmse_cc_vaf_all_offset_cumsumed;
            else
                mean_rmse_cc_vaf_all_offset_cumsumed = 0;
            end
            save_name_package = [save_path 'linear' num2str(linear_model_num) '-nn' num2str(nonlinear_model_num) '-sequence' num2str(sequence_length) '-package.mat'];
            save(save_name_package,'params','save_path','mean_rmse_cc_vaf_all','mean_rmse_cc_vaf_all_abs','mean_rmse_cc_vaf_all_offset','mean_rmse_cc_vaf_all_offset_cumsumed','package_VAF','package_VAF_abs','package_VAF_offset','package_VAF_offset_cumsumed');
    
            %%% 保存每个trial的数据、信息以及每个fold的划分信息，不保存fold具体数据
            decoded_data.params = params;
            save_name_trial = [save_path 'linear' num2str(linear_model_num) '-nn' num2str(nonlinear_model_num) '-sequence' num2str(sequence_length) '-trials.mat'];
            save(save_name_trial,'decoded_data');
    
       end
   end
end


