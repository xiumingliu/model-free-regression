%% Assign y_hat = argmax p(y | x) for unlabeled x inside the region

p_y_x_unlabeled_self = zeros(10, num_unlabeled, 10);
% p_y_x_max = zeros(num_unlabeled, 10);
yhat_x_unlabeled_self = zeros(num_unlabeled, 10);
for y = 0:9
    for i = 1:num_unlabeled
        this_data = data_unlabeled(:, i, y+1);   
        p_y_x_unlabeled_self(1, i, y+1) = bmpdf(this_data, K, model_x_0.mu, model_x_0.w);
        p_y_x_unlabeled_self(2, i, y+1) = bmpdf(this_data, K, model_x_1.mu, model_x_1.w);
        p_y_x_unlabeled_self(3, i, y+1) = bmpdf(this_data, K, model_x_2.mu, model_x_2.w);
        p_y_x_unlabeled_self(4, i, y+1) = bmpdf(this_data, K, model_x_3.mu, model_x_3.w);
        p_y_x_unlabeled_self(5, i, y+1) = bmpdf(this_data, K, model_x_4.mu, model_x_4.w);
        p_y_x_unlabeled_self(6, i, y+1) = bmpdf(this_data, K, model_x_5.mu, model_x_5.w);
        p_y_x_unlabeled_self(7, i, y+1) = bmpdf(this_data, K, model_x_6.mu, model_x_6.w);
        p_y_x_unlabeled_self(8, i, y+1) = bmpdf(this_data, K, model_x_7.mu, model_x_7.w);
        p_y_x_unlabeled_self(9, i, y+1) = bmpdf(this_data, K, model_x_8.mu, model_x_8.w);
        p_y_x_unlabeled_self(10, i, y+1) = bmpdf(this_data, K, model_x_9.mu, model_x_9.w);  

        [~, index] = max(p_y_x_unlabeled_self(:, i, y+1));
        yhat_x_unlabeled_self(i, y+1) = index-1;  
    end
end

% Number of success classified unlabeled data
num_success_unlabeled_self = zeros(10, 1); 
for y = 0:9
    for i = num_adversarial(y+1)+1:num_unlabeled
        if yhat_x_unlabeled_self(i, y+1) == y
            num_success_unlabeled_self(y+1) = num_success_unlabeled_self(y+1) + 1;
        end
    end
end
percentage_success_self = num_success_unlabeled_self./(num_unlabeled - num_adversarial);

num_error_unlabeled_1_self = zeros(10, 1); 
for y = 0:9
    for i = num_adversarial(y+1)+1:num_unlabeled
        if (yhat_x_unlabeled_self(i, y+1) ~= y) 
            num_error_unlabeled_1_self(y+1) = num_error_unlabeled_1_self(y+1) + 1;
        end
    end
end
percentage_error_1_self = num_error_unlabeled_1_self./(num_unlabeled - num_adversarial);

num_error_unlabeled_2_self = zeros(10, 1); 
for y = 0:9
    for i = 1:num_adversarial(y+1)	
        if (~isnan(yhat_x_unlabeled_self(i, y+1))) 
            num_error_unlabeled_2_self(y+1) = num_error_unlabeled_2_self(y+1) + 1;
        end
    end
end
percentage_error_2_self = num_error_unlabeled_2_self./(num_adversarial);

num_error_unlabeled_3_self = zeros(10, 1); 
for y = 0:9
    for i = 1:num_adversarial(y+1)	
        if (~isnan(yhat_x_unlabeled_self(i, y+1))) && (yhat_x_unlabeled_self(i, y+1) ~= y)
            num_error_unlabeled_3_self(y+1) = num_error_unlabeled_3_self(y+1) + 1;
        end
    end
end
percentage_error_3_self = num_error_unlabeled_3_self./(num_adversarial);

% figure; 
% bar(0:9, [num_labeled*ones(10, 1), num_negative_lrt, num_postivie_lrt, num_testing], 'stacked');
% yyaxis right
% plot(0:9, percentage_success, '-k', 'LineWidth', 3);
% legend({'Labeled', 'Inside Region', 'Outside Region', 'Testing', 'Success classfied unlabeled data %'}, 'Location', 'westoutside');
% ylim([0 1]); 

% figure('position', [100, 100, 1000, 600]);
% bar(0:9, [num_labeled*ones(10, 1), num_negative_lrt, num_postivie_lrt, num_testing], 'stacked');
% legend({'Labeled', 'Inside Similar Region', 'Outside Similar Region', 'Testing'}, 'Location', 'westoutside', 'Interpreter', 'latex');
% set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'after_lrt.png'));
% saveas(gcf, fullfile(fpath, 'after_lrt.fig'));

%% Refine p(x | y, l = 1) using p(x | y_hat, l = 0)

data_unlabeled_0_self = [];
data_unlabeled_1_self = [];
data_unlabeled_2_self = [];
data_unlabeled_3_self = [];
data_unlabeled_4_self = [];
data_unlabeled_5_self = [];
data_unlabeled_6_self = [];
data_unlabeled_7_self = [];
data_unlabeled_8_self = [];
data_unlabeled_9_self = [];
data_unlabeled_new_self = []; 

for y = 0:9
    for i = 1:num_unlabeled
        switch yhat_x_unlabeled_self(i, y+1)
            case 0
                data_unlabeled_0_self = [data_unlabeled_0_self, data_unlabeled(:, i, y+1)]; 
            case 1
                data_unlabeled_1_self = [data_unlabeled_1_self, data_unlabeled(:, i, y+1)];
            case 2
                data_unlabeled_2_self = [data_unlabeled_2_self, data_unlabeled(:, i, y+1)];
            case 3
                data_unlabeled_3_self = [data_unlabeled_3_self, data_unlabeled(:, i, y+1)];
            case 4
                data_unlabeled_4_self = [data_unlabeled_4_self, data_unlabeled(:, i, y+1)];
            case 5
                data_unlabeled_5_self = [data_unlabeled_5_self, data_unlabeled(:, i, y+1)];
            case 6
                data_unlabeled_6_self = [data_unlabeled_6_self, data_unlabeled(:, i, y+1)];
            case 7
                data_unlabeled_7_self = [data_unlabeled_7_self, data_unlabeled(:, i, y+1)];
            case 8
                data_unlabeled_8_self = [data_unlabeled_8_self, data_unlabeled(:, i, y+1)];
            case 9
                data_unlabeled_9_self = [data_unlabeled_9_self, data_unlabeled(:, i, y+1)];
            otherwise
                data_unlabeled_new_self = [data_unlabeled_new_self, data_unlabeled(:, i, y+1)];
        end
                
    end        
end

[~, model_x_0_new_self, ~] = mixBernEm([data_labeled(:, :, 1), data_unlabeled_0_self], K);
[~, model_x_1_new_self, ~] = mixBernEm([data_labeled(:, :, 2), data_unlabeled_1_self], K);
[~, model_x_2_new_self, ~] = mixBernEm([data_labeled(:, :, 3), data_unlabeled_2_self], K);
[~, model_x_3_new_self, ~] = mixBernEm([data_labeled(:, :, 4), data_unlabeled_3_self], K);
[~, model_x_4_new_self, ~] = mixBernEm([data_labeled(:, :, 5), data_unlabeled_4_self], K);
[~, model_x_5_new_self, ~] = mixBernEm([data_labeled(:, :, 6), data_unlabeled_5_self], K);
[~, model_x_6_new_self, ~] = mixBernEm([data_labeled(:, :, 7), data_unlabeled_6_self], K);
[~, model_x_7_new_self, ~] = mixBernEm([data_labeled(:, :, 8), data_unlabeled_7_self], K);
[~, model_x_8_new_self, ~] = mixBernEm([data_labeled(:, :, 9), data_unlabeled_8_self], K);
[~, model_x_9_new_self, ~] = mixBernEm([data_labeled(:, :, 10), data_unlabeled_9_self], K);
[~, model_x_u_new_self, ~] = mixBernEm(data_unlabeled_new_self, K);

p_y_0_self = (size(data_unlabeled_0_self, 2)+num_labeled)/(size_training*10);
p_y_1_self = (size(data_unlabeled_1_self, 2)+num_labeled)/(size_training*10);
p_y_2_self = (size(data_unlabeled_2_self, 2)+num_labeled)/(size_training*10);
p_y_3_self = (size(data_unlabeled_3_self, 2)+num_labeled)/(size_training*10);
p_y_4_self = (size(data_unlabeled_4_self, 2)+num_labeled)/(size_training*10);
p_y_5_self = (size(data_unlabeled_5_self, 2)+num_labeled)/(size_training*10);
p_y_6_self = (size(data_unlabeled_6_self, 2)+num_labeled)/(size_training*10);
p_y_7_self = (size(data_unlabeled_7_self, 2)+num_labeled)/(size_training*10);
p_y_8_self = (size(data_unlabeled_8_self, 2)+num_labeled)/(size_training*10);
p_y_9_self = (size(data_unlabeled_9_self, 2)+num_labeled)/(size_training*10);
p_y_u_self = (size(data_unlabeled_new_self, 2))/(size_training*10);

% sum([p_y_0 p_y_1 p_y_2 p_y_3 p_y_4 p_y_5 p_y_6 p_y_7 p_y_8 p_y_9 p_y_u])

p_x_new_self = @(this_data) (p_y_0_self*bmpdf(this_data, K, model_x_0_new_self.mu, model_x_0_new_self.w)+...
    p_y_1_self*bmpdf(this_data, K, model_x_1_new_self.mu, model_x_1_new_self.w)+...
    p_y_2_self*bmpdf(this_data, K, model_x_2_new_self.mu, model_x_2_new_self.w)+...
    p_y_3_self*bmpdf(this_data, K, model_x_3_new_self.mu, model_x_3_new_self.w)+...
    p_y_4_self*bmpdf(this_data, K, model_x_4_new_self.mu, model_x_4_new_self.w)+...
    p_y_5_self*bmpdf(this_data, K, model_x_5_new_self.mu, model_x_5_new_self.w)+...
    p_y_6_self*bmpdf(this_data, K, model_x_6_new_self.mu, model_x_6_new_self.w)+...
    p_y_7_self*bmpdf(this_data, K, model_x_7_new_self.mu, model_x_7_new_self.w)+...
    p_y_8_self*bmpdf(this_data, K, model_x_8_new_self.mu, model_x_8_new_self.w)+...
    p_y_9_self*bmpdf(this_data, K, model_x_9_new_self.mu, model_x_9_new_self.w));


%% Test

p_y_x_testing_self = zeros(10, size_testing, 10);
p_y_x_max_self = zeros(size_testing, 10);
yhat_x_testing_self = zeros(size_testing, 10);
for y = 0:9
    for i = 1:size_testing
        this_data = data_testing(:, i, y+1);   

        p_y_x_testing_self(1, i, y+1) = (p_y_0_self*bmpdf(this_data, K, model_x_0_new_self.mu, model_x_0_new_self.w))/p_x_new_self(this_data);
        p_y_x_testing_self(2, i, y+1) = (p_y_1_self*bmpdf(this_data, K, model_x_1_new_self.mu, model_x_1_new_self.w))/p_x_new_self(this_data);
        p_y_x_testing_self(3, i, y+1) = (p_y_2_self*bmpdf(this_data, K, model_x_2_new_self.mu, model_x_2_new_self.w))/p_x_new_self(this_data);
        p_y_x_testing_self(4, i, y+1) = (p_y_3_self*bmpdf(this_data, K, model_x_3_new_self.mu, model_x_3_new_self.w))/p_x_new_self(this_data);
        p_y_x_testing_self(5, i, y+1) = (p_y_4_self*bmpdf(this_data, K, model_x_4_new_self.mu, model_x_4_new_self.w))/p_x_new_self(this_data);
        p_y_x_testing_self(6, i, y+1) = (p_y_5_self*bmpdf(this_data, K, model_x_5_new_self.mu, model_x_5_new_self.w))/p_x_new_self(this_data);
        p_y_x_testing_self(7, i, y+1) = (p_y_6_self*bmpdf(this_data, K, model_x_6_new_self.mu, model_x_6_new_self.w))/p_x_new_self(this_data);
        p_y_x_testing_self(8, i, y+1) = (p_y_7_self*bmpdf(this_data, K, model_x_7_new_self.mu, model_x_7_new_self.w))/p_x_new_self(this_data);
        p_y_x_testing_self(9, i, y+1) = (p_y_8_self*bmpdf(this_data, K, model_x_8_new_self.mu, model_x_8_new_self.w))/p_x_new_self(this_data);
        p_y_x_testing_self(10, i, y+1) = (p_y_9_self*bmpdf(this_data, K, model_x_9_new_self.mu, model_x_9_new_self.w))/p_x_new_self(this_data);  

        [~, index] = max(p_y_x_testing_self(:, i, y+1));
        yhat_x_testing_self(i, y+1) = index-1;
        
        p_y_x_max_self(i, y+1) = max(p_y_x_testing_self(:, i, y+1));
    end
end

% Number of errors 
num_error_testing_self = zeros(10, 1); 
for y = 0:9
    for i = 1:size_testing
        if yhat_x_testing_self(i, y+1) ~= y
            num_error_testing_self(y+1) = num_error_testing_self(y+1) + 1;
        end
    end
end
percentage_error_testing_self = num_error_testing_self/size_testing;

confusion_matrix_self = eye(10, 10); 
for row = 0:9
    for col = 0:9
        confusion_matrix_self(row+1, col+1) = sum(yhat_x_testing_self(num_adversarial_testing+1:end, col+1) == row); 
    end
end
confusion_matrix_self = confusion_matrix_self/(size_testing - num_adversarial_testing(1));

figure; 
heatmap({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}, ...
    {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}, confusion_matrix_self);
saveas(gcf, fullfile(fpath, 'confusion_matrix_self.png'));
saveas(gcf, fullfile(fpath, 'confusion_matrix_self.fig'));

% figure('position', [100, 100, 600, 600]);
% hold on;
% bar(0:9, [percentage_error_testing_self, percentage_success_self, percentage_error_1_self, percentage_error_2_self, percentage_error_3_self]);
% ylim([0 1])
