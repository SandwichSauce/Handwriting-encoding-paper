function Ahat = nearestSPD(A)
% 计算最近的对称正定矩阵
% A: 输入矩阵
% Ahat: 输出矩阵，最近的对称正定矩阵

% 检查A是否为对称矩阵
if ~isequal(A, A')
    % 如果A不是对称矩阵，则将其转换为对称矩阵
    A = (A + A')/2;
end

% 对对称矩阵进行特征分解
[V, D] = eig(A);

eps = 1e-10;
% 将特征值转换为对角矩阵
d = diag(D);
Dhat = diag(max(d, eps));

% 重新构造矩阵
Ahat = V*Dhat*V';

% 如果重新构造的矩阵不是对称矩阵，则对其进行对称化
if ~isequal(Ahat, Ahat')
    Ahat = (Ahat + Ahat')/2;
end
end