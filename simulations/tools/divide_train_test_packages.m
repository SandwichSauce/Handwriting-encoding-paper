function train_test_package = divide_train_test_packages(trials_data,divide_type)

if strcmp(divide_type,'leave-one-trial-out')
    train_test_package = divide_leave_one_trial_out(trials_data);
elseif strcmp(divide_type,'leave-one-repeat-out')
    train_test_package = divide_leave_one_repeat_out(trials_data);
elseif strcmp(divide_type,'leave-one-repeat-out-types')
    train_test_package = divide_leave_one_repeat_out_types(trials_data);
elseif strcmp(divide_type,'leave-one-char-out')
    train_test_package = divide_leave_one_char_out(trials_data);
end

end


