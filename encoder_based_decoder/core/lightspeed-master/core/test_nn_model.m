function [y] = test_nn_model(x, model)

% x = normalize(x, model.x_mu, model.x_sigma);

pred = nnff(model.nn, x, zeros(size(x,1), model.nn.size(end)));
y = pred.a{end};

% y=bsxfun(@times,y,model.y_sigma);
% y=bsxfun(@plus,y,model.y_mu);


end