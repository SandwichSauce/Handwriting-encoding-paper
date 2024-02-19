function [model,residual] = encode_linear(kin,neu)
%%% kin: kin_dim * time step
%%% neu: neuron_num * time step

% no bias
W_nobias = neu*kin'/(kin*kin');
residual_nobias = neu-W_nobias*kin;
mse_nobias = mean(residual_nobias.*residual_nobias,1)';

% has bias
kin = [ones(1,size(kin,2));kin];
W = neu*kin'/(kin*kin');
residual = neu-W*kin;

mse = mean(residual.*residual,1)';

model.type = 'linear';
model.W = W;
model.residual = residual;
model.mse = mse;

model.W_nobias = W_nobias;
model.residual_nobias = residual_nobias;
model.mse_nobias = mse_nobias;


end

