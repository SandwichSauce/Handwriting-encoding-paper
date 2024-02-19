function plot_traj(predict_test_data,test_data1)

trial_num = numel(predict_test_data.predict_vel_all_trials);
figure();
set(gcf,'Position',[1,41,1536,749.6]);
for trial_no = 1:trial_num
    subplot(5,ceil(trial_num./5),trial_no);
    tmp_target = test_data1.trial_target(trial_no)+1;
    tmp_state = test_data1.groundtruth{tmp_target}.groundtruth_state;
    tmp_pos = [test_data1.groundtruth{tmp_target}.groundtruth_pos.pos_x;test_data1.groundtruth{tmp_target}.groundtruth_pos.pos_y];
    tmp_kin_predict = predict_test_data.predict_vel_all_trials{trial_no};
    tmp_pos_predict = predict_test_data.predict_pos_all_trials{trial_no}';
    tmp_pos_predict_stroke = tmp_pos_predict(:,logical(tmp_state));
    tmp_point_num = numel(tmp_state);
    tmp_pos_predict_original = tmp_pos_predict;
    %%% correct shift
    tmp_pos_predict = tmp_pos_predict - (tmp_pos_predict(:,1) -  tmp_pos(:,1));
    
    for tmp_point_no = 1:tmp_point_num-1
        if tmp_state(tmp_point_no) == 1 && tmp_state(tmp_point_no+1) == 1
            plot(tmp_pos(1,tmp_point_no:tmp_point_no+1),tmp_pos(2,tmp_point_no:tmp_point_no+1),'Color',[0.8,0.8,0.8],'LineWidth',2);
            hold on;
        end
    end


    for tmp_point_no = 1:tmp_point_num-1
        true_start_pos = tmp_pos(:,tmp_point_no);
        current_start_pos = tmp_pos_predict(:,tmp_point_no);
        if tmp_point_no > 1
            if tmp_state(tmp_point_no-1) == 0 && tmp_state(tmp_point_no) == 1
                %%% correct shift
                tmp_pos_predict = tmp_pos_predict - (current_start_pos -  true_start_pos);
            end
        end
        if tmp_state(tmp_point_no) == 1 && tmp_state(tmp_point_no+1) == 1
            plot(tmp_pos_predict(1,tmp_point_no:tmp_point_no+1),tmp_pos_predict(2,tmp_point_no:tmp_point_no+1),'Color',[0.2,0.2,0.2],'LineWidth',2);
        end
    end



end



end

