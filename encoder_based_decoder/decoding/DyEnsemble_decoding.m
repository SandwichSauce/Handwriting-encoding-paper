function [TFC,preKinData] = DyEnsemble_decoding(testNeuroData,TFC)
options = TFC.options;
kin_dim = size(TFC.X_train,1);
model_num = TFC.model_num;
neuron_num = TFC.num_chan;
sf = options.sticking_factor;
resampling_type = options.resamplingScheme;
if options.particle_num ~= 0 
    particle_num = options.particle_num;
end


%%
y = testNeuroData'; 

 

if ~isfield(TFC,'particles')
    % 模型权重初始化，权重平均分配
    model_weights = 1./model_num * ones(1,model_num); 
    % 记录第一个时刻的model weights
    % model_weights_all_time(1,:) = model_weights_all_time(1,:) + model_weights;
    % 各模型的边缘似然
    model_likelihood = ones(1,model_num);

    if options.particle_num ~= 0
        particles = zeros(particle_num,kin_dim) + randn(particle_num,kin_dim);
    else
        particles = TFC.X_train;
        particles = unique(particles','rows')';
        particle_num = size(particles,2);
    end

    TFC.particles = particles;
    TFC.model_weights = model_weights;
    TFC.model_likelihood = model_likelihood;
    TFC.particle_num = particle_num;
else
    particles = TFC.particles;
    model_weights = TFC.model_weights;
    model_likelihood = TFC.model_likelihood;
    particle_num = TFC.particle_num;
end

if options.particle_num ~= 0
    % 粒子用之前训练数据学出来的A进行状态传递
    particles = TFC.A * particles +  mvnrnd(zeros(kin_dim,particle_num),TFC.W);
end


y_decoded_model_particle_all = [];
for model_no = 1:model_num
    tmp_encoder = TFC.model_set{1,model_no};
    tmp_encoder_type = tmp_encoder.type;
    switch tmp_encoder_type
        case 'linear'
            [y_decoded_hasbias,y_decoded_nobias] = decode_linear(particles,tmp_encoder);
            if options.bias_Q
                y_decoded = y_decoded_hasbias;
            else
                y_decoded = y_decoded_nobias;
            end
        case 'poly'
            y_decoded = decode_poly(particles,tmp_encoder);
        case 'nn'
            y_decoded = decode_nn(particles,tmp_encoder);
    end
    % 每一行是一个模型 每一列是一个神经元 每一页是一个粒子
    y_decoded_model_particle_all(model_no,:,:) = y_decoded;
end

y_reshape = reshape(y,[1,neuron_num,1]);
y_error_model_particle_all = y_reshape - y_decoded_model_particle_all;
for model_no = 1:model_num
    %%% 计算一个模型所有粒子的似然
    if options.ifQset
        tmp_Q = TFC.Q_set{model_no}*options.Q_scale + 1e-0 * eye(size(TFC.Q,1));
        tmp_Q = nearestSPD(tmp_Q);
        tmp_model_likelihood = mvnpdf(squeeze(y_error_model_particle_all(model_no,:,:))',zeros(1,neuron_num),tmp_Q);
    else
        tmp_Q = TFC.Q*options.Q_scale + 1e-0 * eye(size(TFC.Q,1));
        tmp_Q = nearestSPD(tmp_Q);
        tmp_model_likelihood = mvnpdf(squeeze(y_error_model_particle_all(model_no,:,:))',zeros(1,neuron_num),tmp_Q);
    end
    likelihood_model_particle(:,model_no) = tmp_model_likelihood;
end

%%% 对于每一个模型，把所有粒子的似然相加，作为这个模型的边缘似然
model_likelihood = sum(likelihood_model_particle,1);
   
%%% 计算模型权重：当前时刻的先验 = 前一时刻的后验.^ff； 当前时刻的后验 = 当前时刻先验 * 似然 再归一化
model_weights_sf = model_weights.^sf;
model_weights_prior = model_weights_sf ./ sum(model_weights_sf); 
model_weights = (model_weights_prior.*model_likelihood) ./ sum(model_weights_prior.*model_likelihood);
%%% zxy add - begin %%%
if sum(isnan(model_weights)) == model_num
    model_weights = 1./model_num * ones(1,model_num); % weights of mixture components
end

for model_no = 1:model_num
    %%% zxy add - begin %%%
    %%% 如果所有粒子的似然都是0，重新平均分配似然
    if length(find(likelihood_model_particle(:,model_no)==0)) == particle_num
        likelihood_model_particle(:,model_no) = 1./particle_num * ones(particle_num,1);
    end
    %%% zxy add - end %%%
    % 归一化似然
    likelihood_model_particle(:,model_no) = likelihood_model_particle(:,model_no)./sum(likelihood_model_particle(:,model_no));  
end


%%% estimation of each models
preKinData_all_model = [];
for model_no = 1:model_num
    preKinData_model = sum(particles .* likelihood_model_particle(:,model_no)',2);
    preKinData_all_model = [preKinData_all_model preKinData_model];
end


preKinData = sum(model_weights .* preKinData_all_model,2);


TFC.preKinData = preKinData;
TFC.particles = particles;
TFC.model_weights = model_weights;
TFC.model_likelihood = model_likelihood;



end

