function [outp] = DyEnsemble_PF(params,test_x,test_y,options)

% est = [];
param_set = {};
% Number of experiments to generate statistical
run_num = 1;            

[kin_dim,test_point_num] = size(test_x);

resampling_type = options.resamplingScheme;
if options.particle_num ~= 0 
    particle_num = options.particle_num;
end
ff = options.forget_factor;

model_num = params.model_num;
neuron_num = params.num_chan;

% MAIN LOOP
% ESS=zeros(run_num,test_point_num);
model_weights_all_time = zeros(test_point_num,model_num); % averaged pai over 30 Monte Carlo runs

clear Error_pf;
clear corr_coef;
%%
if_save_particles_all_time = 0;

x = test_x'; 
y = test_y'; 

% 模型权重初始化，权重平均分配
model_weights = 1./model_num * ones(1,model_num); 
% 记录第一个时刻的model weights
model_weights_all_time(1,:) = model_weights_all_time(1,:) + model_weights;
% 各模型的边缘似然
model_likelihood = ones(1,model_num); 

for run_no = 1:run_num    
    % PERFORM SEQUENTIAL MONTE CARLO
    % INITIALISATION:
    if options.particle_num ~= 0
        if if_save_particles_all_time
            particles_all_time = zeros(test_point_num,particle_num,kin_dim) + randn(test_point_num,particle_num,kin_dim);        % These are the particles for the estimate
        else
            particles = zeros(particle_num,kin_dim) + randn(particle_num,kin_dim);
        end
    else
        if if_save_particles_all_time
            xtrain_particles = options.X_train';
            xtrain_particles_reshape = reshape(xtrain_particles,1,size(xtrain_particles,1),size(xtrain_particles,2));
            particles_all_time = repmat(xtrain_particles_reshape,test_point_num,1,1);
            particle_num = size(options.X_train,2);
        else
            particles = options.X_train;
            particles = unique(particles','rows')';
            particle_num = size(particles,2);
        end
    end
    
    if if_save_particles_all_time
        % One-step-ahead predicted values of the states.
        particle_all_time_transed = zeros(test_point_num,particle_num,kin_dim);  
    end
    % Importance weights.
    likelihood_model_particle_all_time = ones(test_point_num,particle_num,model_num);                   
    % average models for each particle
    likelihood_particle_all_time = zeros(test_point_num,particle_num);      %
  
    estimation_each_model = cell(1,model_num);
    estimation_each_model(:) = {zeros(kin_dim,1)};
    x_estimation_all_time = [];
    for test_point_no = 2:test_point_num
%     for test_point_no = 150:360
        if mod(test_point_no,200)==0
            fprintf('run = %i / %i :  PF : t = %i / %i  \r',run_no,run_num,test_point_no,test_point_num);
            fprintf('\n')    
        end
        
        % PREDICTION STEP:
        if options.particle_num ~= 0
            % 粒子用之前训练数据学出来的A进行状态传递
            if if_save_particles_all_time
                for particle_no = 1:particle_num
                      particle_all_time_transed(test_point_no,particle_no,:) = params.A * squeeze(particles_all_time(test_point_no-1,particle_no,:)) +  mvnrnd(zeros(1,kin_dim),params.W)'; 
                end
            else
                particles = params.A * particles(particle_no,:) +  mvnrnd(zeros(kin_dim,particle_num),params.W);
            end
        else
            % 如果用训练集做粒子，状态就不传递
            if if_save_particles_all_time
                particle_all_time_transed = particles_all_time;
            end
        end
        % kin维度*particle个数
        if if_save_particles_all_time
            particles = squeeze(particle_all_time_transed(test_point_no,:,:))';
        end
        
        y_decoded_model_particle_all = [];
        for model_no = 1:model_num
            tmp_encoder = params.model_set{1,model_no};
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
        
        y_reshape = reshape(y(test_point_no,:),[1,neuron_num,1]);
        y_error_model_particle_all = y_reshape - y_decoded_model_particle_all;


        %%%%%%%%%%%%%%%%%%
        %% 临时验证mse大likelihood小，mse小likelihood大，两种方式，一种是看所有粒子的mse，一种是用真实的GT看mse
%         mse_model = [];
% 
%         for model_no = 1:model_num
%             tmp_residual = squeeze(y_error_model_particle_all(model_no,:,:));
%             tmp_mse = tmp_residual.*tmp_residual;
%             mse_model = [mse_model mean(mean(tmp_mse))];
% 
%         end
% 
%         y_decoded_model_particle_all = [];
%         for model_no = 1:model_num
%             tmp_encoder = params.model_set{1,model_no};
%             tmp_encoder_type = tmp_encoder.type;
%             switch tmp_encoder_type
%                 case 'linear'
%                     [y_decoded_hasbias,y_decoded_nobias] = decode_linear(test_x(:,2),tmp_encoder);
%                     if options.bias_Q
%                         y_decoded = y_decoded_hasbias;
%                     else
%                         y_decoded = y_decoded_nobias;
%                     end
%                 case 'poly'
%                     y_decoded = decode_poly(particles,tmp_encoder);
%                 case 'nn'
%                     y_decoded = decode_nn(particles,tmp_encoder);
%             end
%             % 每一行是一个模型 每一列是一个神经元 每一页是一个粒子
%             y_decoded_model_particle_all(model_no,:,:) = y_decoded;
%         end
%         
%         y_reshape = reshape(y(test_point_no,:),[1,neuron_num,1]);
%         y_error_model_particle_all = y_reshape - y_decoded_model_particle_all;
%         mse = y_error_model_particle_all.*y_error_model_particle_all;
%         mse_model = mean(mse,2);
% 

%%%%%%%%%%



        for model_no = 1:model_num
            %%% 计算一个模型所有粒子的似然
            if options.ifQset
%                 tmp_Q = params.Q_set{model_no}*options.Q_scale;
%                 tmp_Q = nearestSPD(tmp_Q);
                tmp_Q = params.Q_set{model_no}*options.Q_scale + options.diag_scale * eye(size(params.Q,1));
                tmp_Q = tmp_Q + 1e-6*ones(size(tmp_Q));
                tmp_Q = nearestSPD(tmp_Q);
%                 disp(['test_point_no' num2str(test_point_no) ' , model' num2str(model_no)])
                tmp_model_likelihood = mvnpdf(squeeze(y_error_model_particle_all(model_no,:,:))',zeros(1,neuron_num),tmp_Q);
            else
                tmp_Q = params.Q*options.Q_scale + 1e-0 * eye(size(params.Q,1));
                tmp_Q = tmp_Q + 1e-6*ones(size(tmp_Q));
                tmp_Q = nearestSPD(tmp_Q);
                tmp_model_likelihood = mvnpdf(squeeze(y_error_model_particle_all(model_no,:,:))',zeros(1,neuron_num),tmp_Q);
            end
            likelihood_model_particle_all_time(test_point_no,:,model_no) = tmp_model_likelihood;
        end
        
        %%% 对于每一个模型，把所有粒子的似然相加，作为这个模型的边缘似然
        for model_no = 1:model_num
            model_likelihood(model_no) = sum(likelihood_model_particle_all_time(test_point_no,:,model_no)); 
        end
           
        %%% 计算模型权重：当前时刻的先验 = 前一时刻的后验.^ff； 当前时刻的后验 = 当前时刻先验 * 似然 再归一化
        model_weights_ff = model_weights.^ff;
        model_weights_prior = model_weights_ff ./ sum(model_weights_ff); 
        model_weights = (model_weights_prior.*model_likelihood) ./ sum(model_weights_prior.*model_likelihood);
        %%% zxy add - begin %%%
        if sum(isnan(model_weights)) == model_num
            model_weights = 1./model_num * ones(1,model_num); % weights of mixture components
        end
        %%% zxy add - end %%%
        model_weights_all_time(test_point_no,:) = model_weights_all_time(test_point_no,:) + model_weights;
        for model_no = 1:model_num
            %%% zxy add - begin %%%
            %%% 如果所有粒子的似然都是0，重新平均分配似然
            if length(find(likelihood_model_particle_all_time(test_point_no,:,model_no)==0)) == particle_num
                likelihood_model_particle_all_time(test_point_no,:,model_no) = 1./particle_num * ones(1,particle_num);
            end
            %%% zxy add - end %%%
            % 归一化似然
            likelihood_model_particle_all_time(test_point_no,:,model_no) = likelihood_model_particle_all_time(test_point_no,:,model_no)./sum(likelihood_model_particle_all_time(test_point_no,:,model_no));  
            %%% 用模型权重对每个模型所有粒子的似然进行加权，得到所有粒子的似然
            % 注意，这个likelihood_particle_all_time仅用于计算重采样的权重
            % 在计算最后的预测的时候，是先用每个模型所有粒子的似然对粒子加权，得到每个模型预测的速度，
            % 再用模型权重加权每个模型的预测速度
            likelihood_particle_all_time(test_point_no,:) = likelihood_particle_all_time(test_point_no,:) + likelihood_model_particle_all_time(test_point_no,:,model_no) * model_weights(model_no);
        end

        % SELECTION STEP:
        if resampling_type == 1
            outIndex = residualR(1:particle_num,likelihood_particle_all_time(test_point_no,:)');        % Residual resampling.
            particles_all_time(test_point_no,:,:) = particle_all_time_transed(test_point_no,outIndex,:); % Keep particles with
            %%% estimation of each models
            for model_no = 1:model_num
                %%% 根据似然重采样，得到粒子的重采样后的index
                outIndex = multinomialR(1:particle_num,likelihood_model_particle_all_time(test_point_no,:,model_no)');
                particle_all_time_transed_model(test_point_no,:,:) = particle_all_time_transed(test_point_no,outIndex,:); 
                preKinData_model = squeeze(mean(squeeze(particle_all_time_transed_model(test_point_no,:,:)),1))';
                estimation_each_model{model_no} = [estimation_each_model{model_no} preKinData_model];
            end
        elseif resampling_type == 2
            outIndex = systematicR(1:particle_num,likelihood_particle_all_time(test_point_no,:)');      % Systematic resampling.
            particles_all_time(test_point_no,:,:) = particle_all_time_transed(test_point_no,outIndex,:); % Keep particles with
            %%% estimation of each models
            for model_no = 1:model_num
                outIndex = multinomialR(1:particle_num,likelihood_model_particle_all_time(test_point_no,:,model_no)');
                particle_all_time_transed_model(test_point_no,:,:) = particle_all_time_transed(test_point_no,outIndex,:); 
                preKinData_model = squeeze(mean(squeeze(particle_all_time_transed_model(test_point_no,:,:)),1))';
                estimation_each_model{model_no} = [estimation_each_model{model_no} preKinData_model];
            end
        elseif resampling_type == 3
            outIndex = multinomialR(1:particle_num,likelihood_particle_all_time(test_point_no,:)');     % Multinomial resampling.  
            particles_all_time(test_point_no,:,:) = particle_all_time_transed(test_point_no,outIndex,:); % Keep particles with
            %%% estimation of each models
            for model_no = 1:model_num
                outIndex = multinomialR(1:particle_num,likelihood_model_particle_all_time(test_point_no,:,model_no)');
                particle_all_time_transed_model(test_point_no,:,:) = particle_all_time_transed(test_point_no,outIndex,:); 
                preKinData_model = squeeze(mean(squeeze(particle_all_time_transed_model(test_point_no,:,:)),1))';
                estimation_each_model{model_no} = [estimation_each_model{model_no} preKinData_model];
            end
        elseif resampling_type == 0
            %%% estimation of each models
            preKinData_all_model = [];
            for model_no = 1:model_num
                if if_save_particles_all_time
                    preKinData_model = sum(squeeze(likelihood_model_particle_all_time(test_point_no,:,model_no))' .* squeeze(likelihood_model_particle_all_time(test_point_no,:,model_no)),2);
                    estimation_each_model{model_no} = [estimation_each_model{model_no} preKinData_model];
                    preKinData_all_model = [preKinData_all_model preKinData_model];
                else
                    preKinData_model = sum(particles .* squeeze(likelihood_model_particle_all_time(test_point_no,:,model_no)),2);
                    estimation_each_model{model_no} = [estimation_each_model{model_no} preKinData_model];
                    preKinData_all_model = [preKinData_all_model preKinData_model];
                end
            end
            
            if if_save_particles_all_time
                particles_all_time(test_point_no,:,:) = particle_all_time_transed(test_point_no,:,:); % Keep particles with
            end

            preKinData = sum(model_weights .* preKinData_all_model,2);
            x_estimation_all_time = [x_estimation_all_time preKinData];

        end

    end   
   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %-- CALCULATE PERFORMANCE --%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if resampling_type == 0
        x_estimation_all_time = [zeros(size(x_estimation_all_time,1),1) x_estimation_all_time];
        preKinData = x_estimation_all_time;
    else
        preKinData = squeeze(mean(particles_all_time(:,:,:),2))';
    end

    params.estimation = preKinData;
    params.estimation_each_model = estimation_each_model;

    param_set{run_no} = params;

end


% save final results
outp.preKinData = preKinData;

if if_save_particles_all_time
    out.xparticle_pf = particles_all_time;
else
    out.xparticle_pf = particles;
end

outp.no_of_runs = run_num;
outp.param = param_set;
outp.pai_ave = model_weights_all_time;



end