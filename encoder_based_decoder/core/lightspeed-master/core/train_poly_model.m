function [model] = train_poly_model(x,y,order,lambda)

% Polynomial Fitting with L2 Regularization
% input x:      N*M1 vector
%       y:      N*M2 target vector
%       order:  Polynomial order
%       lambda: lambda/2*L2
%       type:   basis function
%                   'Poly' - Polynomial;
%                   'Gaus' - Gaussian;
%                   'Sigm' - Sigmoidal;
% output w:     M1*M2 weight vector 
% Example:
% N=500;
% x = randn(N,2);
% y = zeros(N,3);
% 
% for i=1:N
%     y(i) = 3*x(i,1)+2*x(i,2)^2+1;
% end
% model = train_poly_model(x,y,2,0.5);
% pred = test_poly_model(x,model);

if nargin < 4
    lambda = 0.1;
end

N = size(x,1);      % Sample Count
D = order;          % Order
M1 = size(x,2);
M2 = size(y,2); 
w = zeros(D*M1,M2);


X = ones(N,1);
X = [X x];

% X = x;

w = inv(X'*X+lambda*eye(D))*X'*y;

% H=y*X'*inv(X*X');

e = y - X * w;
m = mean(e,1);
q = cov(e);


model.type = 'poly';
model.m = m;
model.w = w;
model.q = q;
model.order = order;
model.lambda = lambda;


end