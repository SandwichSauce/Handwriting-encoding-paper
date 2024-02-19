function [neu_decoded,neu_decoded_nobias] = decode_linear(kin,model)


% no bias
neu_decoded_nobias = model.W_nobias * kin;


% has bias
kin = [ones(1,size(kin,2));kin];
W = model.W;
neu_decoded = W*kin;


end

