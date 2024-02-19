function [mean_rmse_cc_vaf] = cal_RMSE_CC_VAF_2D(X_test,X_predict)
    mse_cc_vaf = [];
    Q_test = X_test;
    Q_predict = X_predict;
    MSE_vx = mse(Q_test(1,:)-Q_predict(1,:));
    MSE_vy = mse(Q_test(2,:)-Q_predict(2,:));
    RMSE_vx = sqrt(mse(Q_test(1,:)-Q_predict(1,:)));
    RMSE_vy = sqrt(mse(Q_test(2,:)-Q_predict(2,:)));
    CC_vx = corrcoef(Q_test(1,:),Q_predict(1,:));
    CC_vy = corrcoef(Q_test(2,:),Q_predict(2,:));
    VAF_vx = cal_VAF(Q_test(1,:),Q_predict(1,:));
    VAF_vy = cal_VAF(Q_test(2,:),Q_predict(2,:));
    if isnan(CC_vx)
        CC_vx = [0,0;0,0];
    end
    if isnan(CC_vy)
        CC_vy = [0,0;0,0];
    end
    if isnan(VAF_vx)
        VAF_vx = 0;
    end
    if isnan(VAF_vy)
        VAF_vy = 0;
    end
    mse_cc_vaf = [RMSE_vx RMSE_vy CC_vx(1,2) CC_vy(1,2) VAF_vx VAF_vy];
%     rmse_cc_vaf = [sqrt(MSE_vx) CC_vx(1,2) VAF_vx];

    mean_rmse_cc_vaf(1:2) = mse_cc_vaf(1:2);
    mean_rmse_cc_vaf(3) = sqrt((MSE_vx+MSE_vy)./2);
    mean_rmse_cc_vaf(4:5) = mse_cc_vaf(3:4);
    mean_rmse_cc_vaf(6) = mean(mse_cc_vaf(3:4));
    mean_rmse_cc_vaf(7:8) = mse_cc_vaf(5:6);
    mean_rmse_cc_vaf(9) = mean(mse_cc_vaf(5:6));
end

