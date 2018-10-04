%% Test

p_y_x_testing_supervised = zeros(10, size_testing, 10);
p_y_x_max_supervised = zeros(size_testing, 10);
yhat_x_testing_supervised = zeros(size_testing, 10);
for y = 0:9
    for i = 1:size_testing
        this_data = data_testing(:, i, y+1);   

        p_y_x_testing_supervised(1, i, y+1) = (p_y_0*bmpdf(this_data, K, model_x_0.mu, model_x_0.w))/p_x_new(this_data);
        p_y_x_testing_supervised(2, i, y+1) = (p_y_1*bmpdf(this_data, K, model_x_1.mu, model_x_1.w))/p_x_new(this_data);
        p_y_x_testing_supervised(3, i, y+1) = (p_y_2*bmpdf(this_data, K, model_x_2.mu, model_x_2.w))/p_x_new(this_data);
        p_y_x_testing_supervised(4, i, y+1) = (p_y_3*bmpdf(this_data, K, model_x_3.mu, model_x_3.w))/p_x_new(this_data);
        p_y_x_testing_supervised(5, i, y+1) = (p_y_4*bmpdf(this_data, K, model_x_4.mu, model_x_4.w))/p_x_new(this_data);
        p_y_x_testing_supervised(6, i, y+1) = (p_y_5*bmpdf(this_data, K, model_x_5.mu, model_x_5.w))/p_x_new(this_data);
        p_y_x_testing_supervised(7, i, y+1) = (p_y_6*bmpdf(this_data, K, model_x_6.mu, model_x_6.w))/p_x_new(this_data);
        p_y_x_testing_supervised(8, i, y+1) = (p_y_7*bmpdf(this_data, K, model_x_7.mu, model_x_7.w))/p_x_new(this_data);
        p_y_x_testing_supervised(9, i, y+1) = (p_y_8*bmpdf(this_data, K, model_x_8.mu, model_x_8.w))/p_x_new(this_data);
        p_y_x_testing_supervised(10, i, y+1) = (p_y_9*bmpdf(this_data, K, model_x_9.mu, model_x_9.w))/p_x_new(this_data);  

        [~, index] = max(p_y_x_testing_supervised(:, i, y+1));
        yhat_x_testing_supervised(i, y+1) = index-1;
        
        p_y_x_max_supervised(i, y+1) = max(p_y_x_testing_supervised(:, i, y+1));
    end
end

% Number of errors 
num_error_testing_supervised = zeros(10, 1); 
for y = 0:9
    for i = 1:size_testing
        if yhat_x_testing_supervised(i, y+1) ~= y
            num_error_testing_supervised(y+1) = num_error_testing_supervised(y+1) + 1;
        end
    end
end
percentage_error_testing_supervised = num_error_testing_supervised/size_testing;

figure('position', [100, 100, 600, 600]);
hold on;
bar(0:9, [percentage_error_testing_supervised]);
ylim([0 1])
