function [temp_model_set,temp_param_m] = delete_nan_models(temp_model_set,temp_param_m)

nan_model_tag = zeros(1,numel(temp_model_set));
group_no = 1;

% jugde if there is nan, if so, delete this model
for model_no = 1:size(temp_model_set,2)
    nan_model_tag(group_no,model_no) = 1;
    previous_nan_model_num = numel(find(nan_model_tag(group_no,:)==-1));
    new_model_no = model_no - previous_nan_model_num;
    if new_model_no > size(temp_model_set,2)
        break;
    end
    temp_model = temp_model_set{1,new_model_no};
    if isequal(temp_model.type,'linear') || isequal(temp_model.type,'poly')
        if isnan(temp_model.W(1,1))
            nan_model_tag(group_no,model_no) = -1;
            temp_model_set(new_model_no) = [];
            fileds = fieldnames(temp_param_m);
            for i=1:length(fileds)
                k = fileds(i);
                key = k{1};
                if strcmp(key,'model_assign_table')
                    value = temp_param_m.(key);
                    value(new_model_no,:) = [];
                    temp_param_m.(key) = value;
                elseif strcmp(key,'total_err_iter') || strcmp(key,'best_iter_no') || strcmp(key,'min_total_err')
                    continue;
                elseif strcmp(key,'model_assign_table_time')
                    value = temp_param_m.(key);
                    value_num = numel(value);
                    for value_no = 1:value_num
                        tmp_value = value{1,value_no};
                        tmp_value(new_model_no,:) = [];
                        value{1,value_no} = tmp_value;
                    end
                    temp_param_m.(key) = value;
                else
                    value = temp_param_m.(key);
                    value(new_model_no) = [];
                    temp_param_m.(key) = value;
                end
            end
        end
    elseif isequal(temp_model.type,'nn')
        if isnan(temp_param_m.Q_set{1,new_model_no}(1,1))
            nan_model_tag(group_no,model_no) = -1;
            temp_model_set(new_model_no) = [];
            fileds = fieldnames(temp_param_m);
            for i=1:length(fileds)
                k = fileds(i);
                key = k{1};
                if strcmp(key,'model_assign_table')
                    value = temp_param_m.(key);
                    value(new_model_no,:) = [];
                    temp_param_m.(key) = value;
                elseif strcmp(key,'total_err_iter') ||strcmp(key,'best_iter_no')
                    continue;
                elseif strcmp(key,'model_assign_table_time')
                    value = temp_param_m.(key);
                    value_num = numel(value);
                    for value_no = 1:value_num
                        tmp_value = value{1,value_no};
                        tmp_value(new_model_no,:) = [];
                        value{1,value_no} = tmp_value;
                    end
                    temp_param_m.(key) = value;
                else
                    value = temp_param_m.(key);
                    value(new_model_no) = [];
                    temp_param_m.(key) = value;
                end
            end
        end
    end
end
temp_param_m.nan_model_tag = nan_model_tag;

end

