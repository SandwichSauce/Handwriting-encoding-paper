function [model,residual] = encode_poly(kin,neu,order,bias)

x = kin';
y = neu';


lambda = 0.1;


N = size(x,1); % Sample Count
if bias
    D = order+1; % Order+1(bias)
else
    D = order;
end

M1 = size(x,2); % kin_dim
M2 = size(y,2); % neuron_num
W = zeros(D*M1,M2); 


xx = reshape(x,[N*M1,1]);
X = ones(N*M1,D); 
if bias
    for i = 2:D
        X(:,i) = xx.*X(:,i-1);
    end
else
    for i = 1:D
        X(:,i) = xx.*X(:,i);
    end
end
X = reshape(X,[N,M1,D]); 
X = reshape(X,[N,M1*D]); 


W = inv(X'*X+lambda*eye(D*M1))*X'*y;
residual = (y-X*W)';
mse = mean(residual.*residual,1)';

model.type = 'poly';
model.order = order;
model.lambda = lambda;
model.W = W;
model.residual = residual;
model.mse = mse;
model.bias = bias;


end