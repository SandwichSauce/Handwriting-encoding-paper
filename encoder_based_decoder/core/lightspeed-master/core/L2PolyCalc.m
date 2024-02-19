function y = L2PolyCalc(w,x)
% Calculate the result of polynomial.
% input     w: M*1 weight vector
%           x: N*1 vector
% output    y: N*1 output vector


N = size(x,1);      % Sample Count
M = size(w,1);      % Order

X = zeros(N,M);
for i=1:N
    for j=0:M-1
        X(i,j+1) = x(i)^j;
    end
end
y = X*w;

end