function [outp]=Decode_DyLearn(trainNeuroData, trainKinData, testNeuroData, testKinData, options)
% training
if options.bias_kin
    trainKinData_bias = [ones(1,size(trainKinData,2));trainKinData];
else
    trainKinData_bias = trainKinData;
end
Z = trainNeuroData(:,2:end);
X0 = trainKinData_bias(:,1:end-1);
X1 = trainKinData_bias(:,2:end);

A = X1*X0'/((X0*X0'));
H = Z*X1'/(X1*X1');
W = cov([(X1-A*X0)';(X1-A*X0)']);
Q = cov([(Z-H*X1)';(Z-H*X1)']);

params.A = A;
params.H = H;
params.W = W;
params.Q = Q;

% test

neuron_num = size(trainNeuroData,1);

Q_set = [];
model_set = [];
param_m = [];

nan_model_tag = zeros(size(options.encoder_type));

for group_no = 1:options.group_num
    options.group_no = group_no;
    options.group_model_num = length(find(options.encoder_type(group_no,:)~=0));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % generate_model_method_type: 0-learning; 1-random; 2-sequential; 3-speed;
    %                             4-velocity direction; 5-velocity direction;
    %                             6-kin cluster; 7-fr cluster; 
    %                             8-neuron dropout; 9-weight perturbation; 
    generate_model_method_type =  options.generate_model_method_type;
    switch generate_model_method_type
        case 0 % 0-learning;
            [temp_model_set, temp_param_m] = generate_model(trainKinData,trainNeuroData,options.group_model_num,options);
        case 1 % 1-random;
            [temp_model_set, temp_param_m] = generate_model_random_sequential(trainKinData,trainNeuroData,options.group_model_num,options);
        case 2 % 2-sequential;
            [temp_model_set, temp_param_m] = generate_model_random_sequential(trainKinData,trainNeuroData,options.group_model_num,options);
        case 3 % 3-speed;
            [temp_model_set, temp_param_m] = generate_model_speed_direction(trainKinData,trainNeuroData,options.group_model_num,options);
        case 4 % 4-velocity direction;
            [temp_model_set, temp_param_m] = generate_model_speed_direction(trainKinData,trainNeuroData,options.group_model_num,options);
        case 5 % 5-position direction;
            [temp_model_set, temp_param_m] = generate_model_speed_direction(trainKinData,trainNeuroData,options.group_model_num,options);
        case 6 % 6-kin cluster;
            [temp_model_set, temp_param_m] = generate_model_kin_fr_cluster(trainKinData,trainNeuroData,options.group_model_num,options);
        case 7 % 7-fr cluster;
            [temp_model_set, temp_param_m] = generate_model_kin_fr_cluster(trainKinData,trainNeuroData,options.group_model_num,options);
        case 8 % 8-neuron dropout;
        case 9 % 9-weight perturbation;
    end
    % jugde if there is nan, if so, delete this model
    for model_no = 1:size(temp_model_set,2)
        nan_model_tag(group_no,model_no) = 1;
        previous_nan_model_num = numel(find(nan_model_tag(group_no,:)==-1));
        new_model_no = model_no - previous_nan_model_num;
        if new_model_no > size(temp_model_set,2)
            break;
        end
        temp_model = temp_model_set{1,new_model_no};
        if isequal(temp_model.type,'linear') || isequal(temp_model.type,'poly')
            if isnan(temp_model.W(1,1))
                nan_model_tag(group_no,model_no) = -1;
                temp_model_set(new_model_no) = [];
                fileds = fieldnames(temp_param_m);
                for i=1:length(fileds)
                    k = fileds(i);
                    key = k{1};
                    if strcmp(key,'model_assign_table')
                        value = temp_param_m.(key);
                        value(new_model_no,:) = [];
                        temp_param_m.(key) = value;
                    elseif strcmp(key,'total_err_iter') ||strcmp(key,'best_iter_no') ||strcmp(key,'min_total_err')
                        continue;
                    elseif strcmp(key,'model_assign_table_time')
                        value = temp_param_m.(key);
                        value_num = numel(value);
                        for value_no = 1:value_num
                            tmp_value = value{1,value_no};
                            tmp_value(new_model_no,:) = [];
                            value{1,value_no} = tmp_value;
                        end
                        temp_param_m.(key) = value;
                    else
                        value = temp_param_m.(key);
                        value(new_model_no) = [];
                        temp_param_m.(key) = value;
                    end
                end
            end
        elseif isequal(temp_model.type,'nn')
            if isnan(temp_param_m.Q_set{1,new_model_no}(1,1))
                nan_model_tag(group_no,model_no) = -1;
                temp_model_set(new_model_no) = [];
                fileds = fieldnames(temp_param_m);
                for i=1:length(fileds)
                    k = fileds(i);
                    key = k{1};
                    if strcmp(key,'model_assign_table')
                        value = temp_param_m.(key);
                        value(new_model_no,:) = [];
                        temp_param_m.(key) = value;
                    elseif strcmp(key,'total_err_iter') ||strcmp(key,'best_iter_no') ||strcmp(key,'min_total_err')
                        continue;
                    elseif strcmp(key,'model_assign_table_time')
                        value = temp_param_m.(key);
                        value_num = numel(value);
                        for value_no = 1:value_num
                            tmp_value = value{1,value_no};
                            tmp_value(new_model_no,:) = [];
                            value{1,value_no} = tmp_value;
                        end
                        temp_param_m.(key) = value;
                    else
                        value = temp_param_m.(key);
                        value(new_model_no) = [];
                        temp_param_m.(key) = value;
                    end
                end
            end
        end
    end
    model_set = [model_set temp_model_set];
    param_m{group_no} = temp_param_m;
    Q_set = [Q_set temp_param_m.Q_set];
end

model_num = size(model_set,2);

params.Q_set = Q_set;
params.nan_model_tag = nan_model_tag;
params.num_chan = neuron_num;
params.num_mask = size(X0,1);
params.model_num = model_num;
params.model_set = model_set;
params.param_m = param_m;
options.params = params;
options.X_train = trainKinData;

if options.if_train_as_test
    %%% use training set to test
%     [outp] = DyEnsemble_PF(params,trainKinData,trainNeuroData,options);
    [outp] = DyEnsemble_PF(outp.param{1,1},X_train,Y_train,options);
else
    [outp] = DyEnsemble_PF(params,testKinData,testNeuroData,options);
end



end
