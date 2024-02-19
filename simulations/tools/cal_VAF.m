%% function for calculate 1 dimension VAF
function [VAF] = cal_VAF(X,X_est)

%%% 算法一：用var相当于减了均值
% VAF = var(X-X_est) / var(X);
% VAF = 1-VAF;
% 
% if VAF < 0
%     VAF = 0;
% end

%%% 算法二：R2=1-（残差-残差均值）的平方和./总体的平方和 与算法一等价
% X_mean = mean(X);
% residual = X-X_est;
% residual_mean = mean(residual);
% R2 = 1 - sum((X-X_est-residual_mean).^2) ./ sum((X-X_mean).^2);


%%% 算法三：R2=1-残差的平方和./总体的平方和
X_mean = mean(X);
VAF = 1 - sum((X-X_est).^2) ./ sum((X-X_mean).^2);

VAF(isnan(VAF)) = 0;
VAF(find(VAF<0)) = 0;

% if VAF < 0
%     VAF = 0;
% end

end

