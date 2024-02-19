clc;clear;close all;
base_path = 'D:\OneDrive - zju.edu.cn\ZXY\handwriting\handwriting encoding paper\';
load([base_path 'simulations\simulated data\2022-06-10-S1-simulated-snr20.mat']);
params = trials_data.params;
%% encoding config 遍历所有的解码模型和不同线性非线性模型数的配置
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
%            decoder_linear_nonlinear_list = [2,12,0];

    
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
        
            save_path = [base_path 'workspace\' cue_type '\' sort_type '\' divide_type '\encoding_paper\simulation\' params.short_date_session '\' decoder_type '\DyLearn_model_data\' ];
          
            if exist(save_path,'dir')==0
                mkdir(save_path);
            end
            
            params.save_path = save_path;
            trials_data.params = params;
            
            %% 对所有{训练，测试}做编码
%                 encode_pack_list = 1:numel(train_test_package.train_data);
            encode_pack_list = 1; 
            params.encode_pack_list = encode_pack_list;
            trials_data.params = params;
            if params.if_use_bi_decoder
                [train_test_package] = encode_train_test_packages_binary_DyLearn(train_test_package,trials_data,encoder_type,encode_pack_list);
            end
            train_test_package.params = params;
       end
   end
end



        