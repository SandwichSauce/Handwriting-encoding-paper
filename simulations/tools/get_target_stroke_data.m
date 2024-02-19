function target_stroke_data = get_target_stroke_data(target_stroke_list,trial_data,params)

%     target_stroke_list = {'h','s','p','n','d','t','z','g'};

    %%% get spike and velocity for each target stroke
    %%% 找到汉字中所有相同笔画对应的神经信号和速度，比如所有的横对应的数据，所有的竖对应的数据
    all_target_stroke_spike = [];
    all_target_stroke_spike_cell = [];
    all_target_stroke_spike_average = [];
    all_target_stroke_velocity = [];
    all_target_stroke_velocity_cell = [];
    all_target_stroke_char_lable = [];

    target_stroke_num = numel(target_stroke_list);
    for target_stroke_no = 1:target_stroke_num
        target_stroke = target_stroke_list{1,target_stroke_no};
        target_stroke_spike = [];
        target_stroke_spike_cell = [];
        target_stroke_spike_average = [];
        target_stroke_velocity = [];
        target_stroke_velocity_cell = [];
        target_stroke_char_label = [];
        trial_num = numel(trial_data);
        for trial_no = 1:trial_num
            tmp_trial_data = trial_data{1,trial_no};
            target_stroke_index = ismember(tmp_trial_data.stroke_table,target_stroke);
            tmp_target_stroke_spike = tmp_trial_data.single_stroke_spike(target_stroke_index);
            tmp_target_stroke_PVA = tmp_trial_data.single_stroke_PVA(target_stroke_index);
            if isempty(tmp_target_stroke_PVA)
                tmp_target_stroke_velocity = tmp_target_stroke_PVA;
            else
                tmp_num = numel(tmp_target_stroke_PVA);
                tmp_target_stroke_velocity = [];
                for tmp_no = 1:tmp_num
                    tmp_PVA = tmp_target_stroke_PVA{1,tmp_no};
                    tmp_vel = tmp_PVA(3:4,:);
                    tmp_target_stroke_velocity{1,tmp_no} = tmp_vel;
                end
            end
            if isempty(tmp_target_stroke_spike)
                continue;
            end

            % 这里留一个口子，可以得到每一个相同笔画的spike和velocity
            target_stroke_spike_cell = [target_stroke_spike_cell tmp_target_stroke_spike];
            target_stroke_velocity_cell = [target_stroke_velocity_cell tmp_target_stroke_velocity];

            same_stroke_num = numel(tmp_target_stroke_spike);
            for same_stroke_no = 1:same_stroke_num
                target_stroke_spike = [target_stroke_spike tmp_target_stroke_spike{1,same_stroke_no}];
                target_stroke_velocity = [target_stroke_velocity tmp_target_stroke_velocity{1,same_stroke_no}];

                if params.if_label_type_is_cell
                    target_stroke_char_label = [target_stroke_char_label tmp_trial_data.label{1,1}];
                else
                    target_stroke_char_label = [target_stroke_char_label tmp_trial_data.label];
                end
                target_stroke_spike_average = [target_stroke_spike_average mean(tmp_target_stroke_spike{1,same_stroke_no},2)];
            end
        end
        all_target_stroke_spike{target_stroke_no} = target_stroke_spike;
        all_target_stroke_velocity{target_stroke_no} = target_stroke_velocity;
        all_target_stroke_char_lable{target_stroke_no} = target_stroke_char_label;
        all_target_stroke_spike_cell{target_stroke_no} = target_stroke_spike_cell;
        all_target_stroke_velocity_cell{target_stroke_no} = target_stroke_velocity_cell;
        all_target_stroke_spike_average{target_stroke_no} = target_stroke_spike_average;

    end
    
    target_stroke_data.spike = all_target_stroke_spike;
    target_stroke_data.spike_average = all_target_stroke_spike_average;
    target_stroke_data.spike_cell = all_target_stroke_spike_cell;
    target_stroke_data.velocity = all_target_stroke_velocity;    
    target_stroke_data.velocity_cell = all_target_stroke_velocity_cell;
    target_stroke_data.char_label = all_target_stroke_char_lable;
    
    
end

