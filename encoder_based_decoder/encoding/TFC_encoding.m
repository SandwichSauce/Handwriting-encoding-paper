function [TFC] = TFC_encoding(trainNeuroData,trainKinData)

options = get_TFC_options();

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

TFC.A = A;
TFC.H = H;
TFC.W = W;
TFC.Q = Q;

neuron_num = size(trainNeuroData,1);

Q_set = [];
model_set = [];
param_m = [];


for group_no = 1:options.group_num
    options.group_no = group_no;
    options.group_model_num = length(find(options.encoder_type(group_no,:)~=0));
    
    [temp_model_set,temp_param_m] = generate_model(trainKinData,trainNeuroData,options.group_model_num,options);
    [temp_model_set,temp_param_m] = delete_nan_models(temp_model_set,temp_param_m);
    
    model_set = [model_set temp_model_set];
    param_m{group_no} = temp_param_m;
    Q_set = [Q_set temp_param_m.Q_set];
end

model_num = size(model_set,2);

TFC.Q_set = Q_set;
TFC.nan_model_tag = temp_param_m.nan_model_tag;
TFC.num_chan = neuron_num;
TFC.model_num = model_num;
TFC.model_set = model_set;
TFC.param_m = param_m;
TFC.X_train = trainKinData;
TFC.options = options;


end

