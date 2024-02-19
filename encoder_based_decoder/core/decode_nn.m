function [neu_decoded] = decode_nn(kin,model)


x = kin';


pred = nnff(model.nn, x, zeros(size(x,1), model.nn.size(end)));
neu_decoded = pred.a{end}';


end