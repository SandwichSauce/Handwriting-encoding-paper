function [model_set,param_m] = generate_model(trainKinData,trainNeuroData,model_num,options)

X = trainKinData;
Y = trainNeuroData;

original_single_model = encode_linear(X,Y);
if options.bias_mse
    original_single_mse = original_single_model.mse;
else
    original_single_mse = original_single_model.mse_nobias;
end
original_single_mse = smooth(original_single_mse,options.err_smooth_winsize);

linear_select_index = find(original_single_mse' <= options.linear_prior_threshold);
linear_left_index = find(original_single_mse' > options.linear_prior_threshold);
X_linear_select = X(:,linear_select_index); 
Y_linear_select = Y(:,linear_select_index);


data_size = size(trainNeuroData,2);

model_set = [];
model_set_time = [];
Q_set = [];
mse_local_set = [];
mse_global_set = [];
mse_global_set_smooth = [];

param_m = {};

max_iter_num = options.max_iter_num;
min_iter_num = options.min_iter_num;
keep_percent = options.keep_percent;
err_smooth_winsize = options.err_smooth_winsize;

encoder_type_list = options.encoder_type(options.group_no,:);


%%% get original single model mse
encoder_type = encoder_type_list(1);
switch encoder_type
    case 1
        original_single_model = encode_linear(X,Y);
        original_single_residual = original_single_model.residual;
        if options.bias_mse
            original_single_mse = original_single_model.mse;
        else
            original_single_mse = original_single_model.mse_nobias;
        end
        
        original_single_Q_bias = cov(original_single_model.residual');
        original_single_Q_nobias = cov(original_single_model.residual_nobias');
        original_single_Q_bias_diff = original_single_Q_bias - original_single_Q_nobias;
        if options.bias_Q
            original_single_Q = original_single_Q_bias;
        else
            original_single_Q = original_single_Q_nobias;
        end
    case 2
        bias = options.bias_mse;
        order = 2;
        original_single_model = encode_poly(X,Y,order,bias);
        original_single_residual = original_single_model.residual;
        original_single_mse = original_single_model.mse;
        original_single_Q = cov(original_single_model.residual');
    case 3
        original_single_model = encode_nn(X,Y,options);
        original_single_residual = original_single_model.residual;
        original_single_mse = original_single_model.mse;
        original_single_Q = cov(original_single_model.residual');
end
original_single_total_error = nanmean(original_single_mse);
% subset assignment

model_assign_table = zeros(model_num,data_size);

for model_no = 1:model_num
    idx = randperm(data_size);
    idx = idx(1:round(data_size*keep_percent));
    model_assign_table(model_no,idx)=1;
end

if options.linear_prior_threshold > 0
    model_assign_table(1,linear_select_index) = 1;
    model_assign_table(1,~linear_select_index) = 0;
    model_assign_table(2:end,linear_select_index) = 0;
end

model_assign_table_time{1} = model_assign_table;
param_m.total_err_iter(1) = original_single_total_error;
for model_no = 1:model_num
    model_set_time{model_no}{1} = original_single_model;
    Q_set_time{model_no}{1} = original_single_Q;
end

% matching and reassignment
for iter_no = 1:max_iter_num
    % fitting
    for model_no = 1:model_num
        idx = find(model_assign_table(model_no,:) == 1);
        xx = X(:,idx);
        yy = Y(:,idx);
        
        encoder_type = encoder_type_list(model_no);
        
        switch encoder_type
            case 1
                tmp_model = encode_linear(xx,yy);
                residual_local = tmp_model.residual;
                if options.bias_mse
                    mse_local = tmp_model.mse;
                else
                    mse_local = tmp_model.mse_nobias;
                end
                Q_bias = cov(tmp_model.residual');
                Q_nobias = cov(tmp_model.residual_nobias');
                Q_bias_diff = Q_bias - Q_nobias;
                if options.bias_Q
                    Q = Q_bias;
                else
                    Q = Q_nobias;
                end
                [Y_decoder_hasbias,Y_decoder_nobias] = decode_linear(X,tmp_model);
                if options.bias_mse
                    Y_decoded = Y_decoder_hasbias;
                else
                    Y_decoded = Y_decoder_nobias;
                end
                residual_global = Y - Y_decoded;
                mse_global = mean(residual_global.*residual_global,1)';
            case 2
                bias = options.bias_mse;
                order = 2;
                tmp_model = encode_poly(xx,yy,order,bias);
                residual_local = tmp_model.residual;
                mse_local = tmp_model.mse;
                Q = cov(tmp_model.residual');
                Y_decoded = decode_poly(X,tmp_model);
                residual_global = Y - Y_decoded;
                mse_global = mean(residual_global.*residual_global,1)';
            case 3
                tmp_model = encode_nn(xx,yy,options);
                residual_local = tmp_model.residual;
                mse_local = tmp_model.mse;
                Q = cov(tmp_model.residual');
                Y_decoded = decode_nn(X,tmp_model);
                residual_global = Y - Y_decoded;
                mse_global = mean(residual_global.*residual_global,1)';
        end
        
        Q_set_time{model_no}{iter_no+1} = Q;
        model_set_time{model_no}{iter_no+1} = tmp_model;

        mse_local_set{model_no} = mse_local;
        mse_global_set(model_no,:) = mse_global;
        mse_global_set_smooth(model_no,:) = smooth(mse_global,err_smooth_winsize);
        
    end
    
    [min_mse_global] = min(mse_global_set,[],1);
    [min_mse_global_set_smooth, best_model_index] = min(mse_global_set_smooth,[],1);
    
    % reassignment    
    model_assign_table = zeros(model_num,data_size);  
    
    for model_no = 1:model_num
        model_assign_table(model_no,find(best_model_index==model_no))=1;
    end
    if options.linear_prior_threshold > 0
        model_assign_table(1,linear_select_index) = 1;
        model_assign_table(2:end,linear_select_index) = 0;
    end

    model_assign_table_time{iter_no+1} = model_assign_table;
    
    param_m.Q_set_time = Q_set_time;
    param_m.model_set_time = model_set_time;
    param_m.model_assign_table_time = model_assign_table_time;
    
    % early stop
    
    total_error_list1 = [];
    total_error_list2 = min_mse_global;

    error_type = 2;
    for model_no = 1:model_num
        total_error_list1 = [total_error_list1;mse_local_set{model_no}];

    end
    if error_type == 1
        total_error = nanmean(total_error_list1);
    elseif error_type == 2
        total_error = nanmean(total_error_list2);
    end
    param_m.total_err_iter(iter_no+1) = total_error;

    if iter_no > min_iter_num
        if (param_m.total_err_iter(iter_no-1) - param_m.total_err_iter(iter_no)) < 0.001 * param_m.total_err_iter(iter_no)
            break;
        end
    end
end

[min_total_err,best_iter_no] = min(param_m.total_err_iter);

model_assign_table = model_assign_table_time{best_iter_no};
for model_no = 1:model_num
    model_set{model_no} = model_set_time{model_no}{best_iter_no};
    Q_set{model_no} = Q_set_time{model_no}{best_iter_no};
end

param_m.model_assign_table = model_assign_table;
param_m.model_set = model_set;
param_m.Q_set = Q_set;
param_m.best_iter_no = best_iter_no;
param_m.min_total_err = min_total_err;



end

