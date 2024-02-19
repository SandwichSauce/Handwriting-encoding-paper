function [neu_decoded] = decode_poly(kin,model)


x = kin';

N = size(x,1);
if model.bias
    D = model.order+1;   
else
    D = model.order;
end
M1 = size(x,2);  
W = model.W;


xx = reshape(x,[N*M1,1]);
X = ones(N*M1,D);
if model.bias
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

neu_decoded = (X*W)';
end