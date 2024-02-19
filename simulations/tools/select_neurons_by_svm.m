function [good_neuron_index,svm_acc_sort,svm_acc_sort_index] = select_neurons_by_svm(trials_data)


neuron_num = size(trials_data.trial_data{1}.spike_bin,1);
Y = trials_data.trial_target'+1;
trial_num = size(Y,1);
acc_set = [];

for i = 1:neuron_num
    X = [];
    for j = 1:trial_num
        X(j,:) = smooth(trials_data.trial_data{j}.spike_bin(i,1:min(trials_data.trial_target_length_list)));
    end
    [pred cm acc] = my_multi_svm(X(1:round(trial_num*2./3),:), Y(1:round(trial_num*2./3),:), X((round(trial_num*2./3)+1):end,:), Y((round(trial_num*2./3)+1):end,:));
    acc_set(i) = acc;
    
end
[svm_acc_sort,svm_acc_sort_index] = sort(acc_set,'descend');
select_neuron_by_svm_threshold = trials_data.params.select_neuron_by_svm_threshold;
good_neuron_index = find(acc_set>select_neuron_by_svm_threshold);

end

