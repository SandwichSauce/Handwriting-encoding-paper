%% function for calculate 1 dimension VAF
function [VAF] = cal_VAF(X,X_est)

%%% �㷨һ����var�൱�ڼ��˾�ֵ
% VAF = var(X-X_est) / var(X);
% VAF = 1-VAF;
% 
% if VAF < 0
%     VAF = 0;
% end

%%% �㷨����R2=1-���в�-�в��ֵ����ƽ����./�����ƽ���� ���㷨һ�ȼ�
% X_mean = mean(X);
% residual = X-X_est;
% residual_mean = mean(residual);
% R2 = 1 - sum((X-X_est-residual_mean).^2) ./ sum((X-X_mean).^2);


%%% �㷨����R2=1-�в��ƽ����./�����ƽ����
X_mean = mean(X);
VAF = 1 - sum((X-X_est).^2) ./ sum((X-X_mean).^2);

VAF(isnan(VAF)) = 0;
VAF(find(VAF<0)) = 0;

% if VAF < 0
%     VAF = 0;
% end

end

