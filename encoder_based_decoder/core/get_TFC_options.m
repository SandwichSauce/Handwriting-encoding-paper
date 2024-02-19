function options = get_TFC_options()


    % k: use k time steps' neural information 
    options.sequence_length = 1;

    % 0: no resampling, 1: residual, 2: systematic sampling, 3: multinomial. 
    options.resamplingScheme = 0; 
    
    % 0: training set as particle set, k: randomly generate k  particles
    options.particle_num = 0;
    
    % 0: same prior ~ 1: posteriori of the previous time step
    options.sticking_factor = 0;

    % 0: x_vel,y_vel, 1: x_pos,y_pos,x_vel,y_vel,x_acc,y_acc
    options.ifkindim6 = 0;
    
    % 1: different models use different Q
    options.ifQset = 1;
    
    % 1: assign data according to biased mse
    options.bias_mse = 0;
    
    % 1: calulate local Q_set according to biased residual
    options.bias_Q = 0;
    options.Q_scale = 1;
    
    % 1: calulate global Q according to biased residual
    options.bias_kin = 0;
    
    % generate_model_method_type: 0-learning; 1-random; 2-sequential; 3-speed;
    %                             4-velocity direction; 5-velocity direction;
    %                             6-kin cluster; 7-fr cluster; 
    %                             8-neuron dropout; 9-weight perturbation; 
    options.generate_model_method_type = 0;

    options.linear_prior_threshold = 0;


    % 1 group: 10 linear
    options.encoder_type = [[1,1,1,1,1,1,1,1,1,1]];

    % 3 groups: 5 linear + 5 poly + 5 nn
%     options.encoder_type = [[1,1,1,1,1];...
%                             [2,2,2,2,2];...
%                             [3,3,3,3,3]];

    % 4 groups: 1 linear + 4 linear + 1 nn + 4 nn
%     options.encoder_type = [[1,0,0,0];...
%                             [1,1,1,1];...
%                             [3,0,0,0];...
%                             [3,3,3,3]];


    options.group_num = size(options.encoder_type,1);
    options.model_num = numel(options.encoder_type);
    options.err_smooth_winsize = 10;
    options.max_iter_num = 300;
    options.min_iter_num = 50;
    options.keep_percent = 0.1;
    
    %%% nn options
    options.hidden_neuorn_num_type = 'preset';   % 'preset' or 'adaptive'
    options.hidden_neuorn_num = 100;
    options.hidden_neuorn_num_times = 10;
    options.learning_rate = 0.1;
    options.scaling_learning_rate = 0.9;
    options.epoch_num = 20;
    options.batch_size = 10;
    options.if_plot = 0;
    
    %%% test options %%%
    options.if_train_as_test = 0;

end

