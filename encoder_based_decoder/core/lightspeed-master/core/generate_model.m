function [W_set,Q_set,model_assign_table] = generate_model(trainKinData,trainNeuroData,model_num,Qs)


X = trainKinData;
Y = trainNeuroData;

data_size = size(trainNeuroData,2);

W_set = [];
Q_set = [];
W_set_time = [];
W_mat = [];
Err_trace = [];
Err_trace_long = [];
Err_set = [];
Err_set_long = [];
Err_set_long_smooth = [];

% model_num = 5;
iter_num = 20;
keep_percent = 0.3;

% subset assignment

model_assign_table = zeros(model_num,data_size);

for k = 1:model_num
    idx = randperm(data_size);
    idx = idx(1:round(data_size*keep_percent));
    model_assign_table(k,idx)=1;
end

% matching and reassignment


for i = 1:iter_num
    
    % fitting
    for k = 1:model_num
        idx = find(model_assign_table(k,:) == 1);
        xx = X(:,idx);
        yy = Y(:,idx);
        
        %加bias，这里不用撤销bias,因为bias加在临时变量xx上
        xx=[ones(1,size(xx,2));xx];
        %W = yy*xx'/(xx*xx');
        W = yy*xx'*inv(xx*xx');
        W_set{k} = W;
        W_set_time{k}{i}=W;
        %Q=cov([(yy-W*xx)';(yy-W*xx)']);
        Q=cov((yy-W*xx)');
        [R,e]=cholcov(Q,0);
        if e~=0
            Q=1*eye(size(yy,1));
            %Q=Qs+rand(size(yy,1));
        end
        Q_set{k}=Q;
        err = mean((W*xx-yy).*(W*xx-yy),1)';
        %加bias
        X=[ones(1,size(X,2));X];
        err_long = mean((W*X-Y).*(W*X-Y),1)';
        %加上bias,再撤销因为是循环生成的
        X=X(2:end,:);
        
        Err_trace{k}(i) = sqrt(err'*err)/size(err,1);
        Err_set{k} = err;
        Err_set_long(k,:) = err_long;
        Err_set_long_smooth(k,:) = smooth(err_long,10);
    end
    
    [min_err] = min(Err_set_long);
    [~, min_idx] = min(Err_set_long_smooth);
    Err_trace_merge(i) = sqrt(min_err*min_err')/data_size;
    
    % reassignment    
    model_assign_table = zeros(model_num,data_size);  
    
    for k = 1:model_num
        model_assign_table(k,find(min_idx==k))=1;
    end
    
    
%     for k = 1:model_num
%         err = Err_set_long(k,:);
%         err = smooth(err,5);
%         err_sort = sort(err);
%         threshold = err_sort(round(data_size*keep_percent));
%         idx = find(err<=threshold);
%         model_assign_table(k,idx)=1;
%     end
    
%     if i <= 6     
%         figure(2);
%         subplot(2,3,i);
%         colors = 'rgbck';
%         x_axis = [0:0.01:10]';
%         for k = 1:min(model_num, 5)
%             y_axis = L2PolyCalc(W_set{k},x_axis);
%             plot(x_axis,y_axis,[colors(k), '-']);
%             hold on;
%             idx = find(model_assign_table(k,:)==1);
%             x_data = X(idx);
%             y_data = Y(idx);
%             plot(x_data,y_data,[colors(k), '.']);
%             hold on;
%         end
%         hold off;
%     end
    
end

