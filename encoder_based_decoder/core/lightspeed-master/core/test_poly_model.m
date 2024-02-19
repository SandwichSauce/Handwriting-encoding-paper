function [y] = test_poly_model(x,model)
% Calculate the result of polynomial.
% input     w: D*M weight vector
%           x: N*M1 vector
% output    y: N*M2 output vector


N = size(x,1);      % Sample Count
D = model.order;      % Order
M1 = size(x,2);      % Number of Features
w = model.w;

X = ones(N,1);
X = [X x];

% X = x; 

y = X*w;
end