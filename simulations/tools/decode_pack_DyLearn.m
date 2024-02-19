function decoded_info = decode_pack_DyLearn(Y_train, X_train, Y_test, X_test, plot_info)
    options = get_DyLearn_options();
    options.plot_info = plot_info;
    if plot_info.if_replace_true_data_with_simulated
        options.err_smooth_winsize = plot_info.params.DyLearn_error_smooth;
        options.keep_percent = 0.2;
        options.bias_mse = 1;
        options.bias_Q = 1;
        options.diag_scale = 0;
    end
    
    %%% 看下这个配置能不能解
%     options.err_smooth_winsize = 1;
%     options.keep_percent = 0.2;
%     options.bias_mse = 1;
%     options.bias_Q = 1;
%     options.diag_scale = 0;

    options.sequence_length = plot_info.sequence_length;
    options.encoder_type = plot_info.encoder_type;
    options.linear_prior_threshold = plot_info.linear_prior_threshold;
    options.group_num = size(options.encoder_type,1);
    options.model_num = numel(options.encoder_type);
    options.short_date_session = plot_info.short_date_session;

    outp = Decode_DyLearn(Y_train, X_train, Y_test, X_test, options);
    X_predict = outp.preKinData;
    
%     DyLearn_TF = outp;
    DyLearn_TF.param{1,1}.nan_model_tag = outp.param{1,1}.nan_model_tag;

    %%% 空间充足
%     DyLearn_TF.param{1,1}.param_m = outp.param{1,1}.param_m;

    %%% 空间不充足
    DyLearn_TF.param{1,1}.param_m{1,1}.total_err_iter = outp.param{1,1}.param_m{1,1}.total_err_iter;
    DyLearn_TF.param{1,1}.param_m{1,1}.model_assign_table_time = outp.param{1,1}.param_m{1,1}.model_assign_table_time;
    DyLearn_TF.param{1,1}.param_m{1,1}.best_iter_no = outp.param{1,1}.param_m{1,1}.best_iter_no;


%     DyLearn_TF.pai_ave = outp.pai_ave;
    DyLearn_TF.model_weights = outp.pai_ave';
    DyLearn_TF.options = options;
    
    decoded_info.X_predict = X_predict;
    decoded_info.TF = DyLearn_TF;
end