function [model,residual] = encode_nn(kin,neu,options)

x = kin';
y = neu';

input_size = size(x,2);
output_size = size(y,2);

if strcmp(options.hidden_neuorn_num_type,'preset')
    hidden_number = options.hidden_neuorn_num;
elseif strcmp(options.hidden_neuorn_num_type,'adaptive')
    hidden_number = options.hidden_neuorn_num_times * input_size;
end

learning_rate = options.learning_rate;
scaling_learning_rate = options.scaling_learning_rate;
epoch_num = options.epoch_num;
batch_size = options.batch_size;
if_plot = options.if_plot;
    

nn                      = nnsetup([input_size hidden_number output_size]);   
nn.activation_function  = 'sigm';
nn.learningRate         = learning_rate;
nn.scaling_learningRate = scaling_learning_rate;
nn.output               = 'linear';                    %  use softmax output
opts.numepochs          = epoch_num;                       %  Number of full sweeps through data
opts.batchsize          = batch_size;                        %  Take a mean gradient step over this many samples
opts.plot               = if_plot;                         %  enable plotting
nn = nntrain(nn, x, y, opts);                        %  nntrain takes validation set as last two arguments (optionally)

pred = nnff(nn,x,zeros(size(x,1),nn.size(end)));
y_decoded = pred.a{end};
residual = (y - y_decoded)';
mse = mean(residual.*residual,1)';

model.type = 'nn';
model.hidden_number = hidden_number;
model.residual = residual;
model.mse = mse;
model.nn = nn;

% model.x_mu = x_mu;
% model.x_sigma = x_sigma;
% model.y_mu = y_mu;
% model.y_sigma = y_sigma;

end