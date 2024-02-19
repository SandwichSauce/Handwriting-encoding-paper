function train_test_package = divide_leave_one_trial_out(trials_data)
%% divide data to train and test - leave one trial out

trial_target = trials_data.trial_target;
trial_data = trials_data.trial_data;

%%% get trial target & unique target & unique char
trial_target_list = trials_data.trial_target;
unique_target_list = unique(trial_target_list);
trial_target_num = numel(trial_target_list);
unique_target_num = numel(unique_target_list);


for trial_target_no = 1:trial_target_num
    tmp_test_data = trial_data(trial_target_no);

    %%% test data
    tmp_data = tmp_test_data;
    test_spike = tmp_data{1,1}.spike_bin;
    if trials_data.params.if_offline_lstm_data
        test_vel = tmp_data{1,1}.vel_kin_bin;
        test_pos = cumsum(test_vel')';
        test_acc = diff(test_vel,[],2);
        test_acc = [test_acc test_acc(end)];
        test_PVA = [test_pos;test_vel;test_acc];
    else
        test_PVA = tmp_data{1,1}.PVA_bin;
    end
    test_stroke_PVA = tmp_data{1,1}.stroke_PVA;
    test_stroke_spike = tmp_data{1,1}.stroke_spike;
    test_break_PVA = tmp_data{1,1}.break_PVA;
    test_break_spike = tmp_data{1,1}.break_spike;
    test_state=tmp_data{1,1}.state_bin;
    test_label=tmp_data{1,1}.label;
    test_target=tmp_data{1,1}.target;
    
    
    %%% 只用分类性能好的神经元
    if trials_data.params.if_select_neurons
        test_spike = test_spike(trials_data.good_neuron_by_svm_list,:);
        test_stroke_spike = test_stroke_spike(trials_data.good_neuron_by_svm_list,:);
        test_break_spike = test_break_spike(trials_data.good_neuron_by_svm_list,:);
    end
    
    %%% 并且用前10个点平滑
    if trials_data.params.if_offline_lstm_data    
        test_spike = movmean(test_spike(trials_data.params.good_neuron_list,:),[10 0],2);
        test_stroke_spike = movmean(test_stroke_spike(trials_data.params.good_neuron_list,:),[10 0],2);
        test_break_spike = movmean(test_break_spike(trials_data.params.good_neuron_list,:),[10 0],2);
    end
    
    
    test_data{trial_target_no}.test_spike=test_spike;
    test_data{trial_target_no}.test_PVA = test_PVA;
    test_data{trial_target_no}.test_stroke_PVA = test_stroke_PVA;
    test_data{trial_target_no}.test_stroke_spike=test_stroke_spike;
    test_data{trial_target_no}.test_break_PVA = test_break_PVA;
    test_data{trial_target_no}.test_break_spike=test_break_spike;
    test_data{trial_target_no}.test_state=test_state;
    test_data{trial_target_no}.test_label=test_label;
    test_data{trial_target_no}.test_target=test_target;

end

for i_target=1:length(unique(trial_target))
    index_target=find(trial_target==i_target-1);
    %%% train data
    tmp_train_data=trial_data;
    tmp_train_data(index_target)=[];
    train_spike=[];
    train_PVA = [];
    train_state=[];
    train_stroke_PVA = [];
    train_stroke_spike=[];
    train_break_PVA = [];
    train_break_spike=[];
    train_state=[];
    
    for i_train =1:length(tmp_train_data)
        tmp_data=tmp_train_data(i_train);
        tmp_spike=tmp_data{1,1}.spike_bin;
        
        if trials_data.params.if_offline_lstm_data
            tmp_vel = tmp_data{1,1}.vel_kin_bin;
            tmp_pos = cumsum(tmp_vel')';
            tmp_acc = diff(tmp_vel,[],2);
            tmp_acc = [tmp_acc tmp_acc(end)];
            tmp_PVA = [tmp_pos;tmp_vel;tmp_acc];
        else
            tmp_PVA = tmp_data{1,1}.PVA_bin;
        end

        tmp_stroke_PVA = tmp_data{1,1}.stroke_PVA;
        tmp_stroke_spike=tmp_data{1,1}.stroke_spike;
        
        tmp_break_PVA = tmp_data{1,1}.break_PVA;
        tmp_break_spike=tmp_data{1,1}.break_spike;
        
        tmp_state=tmp_data{1,1}.state_bin;
        
        train_spike=[train_spike,tmp_spike];
        train_PVA = [train_PVA,tmp_PVA];
        
        train_stroke_PVA = [train_stroke_PVA,tmp_stroke_PVA];
        train_stroke_spike=[train_stroke_spike,tmp_stroke_spike];
        
        train_break_PVA = [train_break_PVA,tmp_break_PVA];
        train_break_spike=[train_break_spike,tmp_break_spike];
        train_state=[train_state,tmp_state];
    end
    
    %%% 只用分类性能好的神经元
    if trials_data.params.if_select_neurons
        train_spike = train_spike(trials_data.good_neuron_by_svm_list,:);
        train_stroke_spike = train_stroke_spike(trials_data.good_neuron_by_svm_list,:);
        train_break_spike = train_break_spike(trials_data.good_neuron_by_svm_list,:);
    end
    
    %%% 并且用前10个点平滑
    if trials_data.params.if_offline_lstm_data    
        train_spike = movmean(train_spike(trials_data.params.good_neuron_list,:),[10 0],2);
        train_stroke_spike = movmean(train_stroke_spike(trials_data.params.good_neuron_list,:),[10 0],2);
        train_break_spike = movmean(train_break_spike(trials_data.params.good_neuron_list,:),[10 0],2);
    end
    
    for i_block=1:length(index_target)
        train_data{index_target(i_block)}.train_spike=train_spike;
        train_data{index_target(i_block)}.train_PVA = train_PVA;
        train_data{index_target(i_block)}.train_stroke_PVA = train_stroke_PVA;
        train_data{index_target(i_block)}.train_stroke_spike=train_stroke_spike;
        train_data{index_target(i_block)}.train_break_PVA = train_break_PVA;
        train_data{index_target(i_block)}.train_break_spike=train_break_spike;
        train_data{index_target(i_block)}.train_state=train_state;
   end
    
end

train_test_package.train_data=train_data;
train_test_package.test_data=test_data;
train_test_package.trial_target=trial_target;

end


