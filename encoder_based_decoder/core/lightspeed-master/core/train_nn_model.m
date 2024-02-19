function [model] = train_nn_model(x,y,hidden_number)

% 
% input x:      N*M1 vector
%       y:      N*M2 target vector
% Example:
% N=5000;
% train_x = randn(N,20);
% [train_x, mu, sigma] = zscore(train_x);
% W = rand(20,30);
% train_y = train_x*W;
% train_y = train_y + rand(N,30)*0.001;
% model = train_nn_model(train_x,train_y,50);
% pred = test_nn_model(train_x,model);
% CC = corrcoef(pred,train_y);
% disp(CC(2,1));

%[x, x_mu, x_sigma] = zscore(x);
% [y, y_mu, y_sigma] = zscore(y);
% x = normalize(x, x_mu, x_sigma);

M1 = size(x,2);
M2 = size(y,2);

nn                      = nnsetup([M1 hidden_number M2]);   
nn.activation_function  = 'sigm';
nn.learningRate         = 0.03;
nn.output               = 'linear';                    %  use softmax output
opts.numepochs          = 10000;                       %  Number of full sweeps through data
opts.batchsize          = 20;                        %  Take a mean gradient step over this many samples
opts.plot               = 0;                         %  enable plotting
nn = nntrain(nn, x, y, opts);                        %  nntrain takes validation set as last two arguments (optionally)

model.type = 'nn';
model.nn = nn;
% model.x_mu = x_mu;
% model.x_sigma = x_sigma;
% model.y_mu = y_mu;
% model.y_sigma = y_sigma;

end