%% LRT, p(x | l = 0) v.s. p(x | l = 1), for unlabeled data

LLR_unlabeled = zeros(num_unlabeled, 10); 

for y = 0:9
for i = 1:num_unlabeled
    this_data = data_unlabeled(:, i, y+1); 
    LLR_unlabeled(i, y+1) = log(bmpdf(this_data, K, model_x_u.mu, model_x_u.w))-...
        log(bmpdf(this_data, K, model_x_l.mu, model_x_l.w)); 
    if isnan(LLR_unlabeled(i, y+1))
        LLR_unlabeled(i, y+1) = kappa; 
    end
end
end

% Number of positive log likelihood ratios
num_postivie_lrt = zeros(10, 1); 
for y = 0:9
    this_lrt = LLR_unlabeled(:, y+1);
    num_postivie_lrt(y+1) = sum(this_lrt >=kappa); 
end
num_negative_lrt = num_unlabeled*ones(10, 1) - num_postivie_lrt;



%% Assign y_hat = argmax p(y | x) for unlabeled x inside the region

p_y_x_unlabeled = zeros(10, num_unlabeled, 10);
% p_y_x_max = zeros(num_unlabeled, 10);
yhat_x_unlabeled = zeros(num_unlabeled, 10);
for y = 0:9
    for i = 1:num_unlabeled
        this_data = data_unlabeled(:, i, y+1);   
        if LLR_unlabeled(i, y+1) < kappa
            p_y_x_unlabeled(1, i, y+1) = bmpdf(this_data, K, model_x_0.mu, model_x_0.w);
            p_y_x_unlabeled(2, i, y+1) = bmpdf(this_data, K, model_x_1.mu, model_x_1.w);
            p_y_x_unlabeled(3, i, y+1) = bmpdf(this_data, K, model_x_2.mu, model_x_2.w);
            p_y_x_unlabeled(4, i, y+1) = bmpdf(this_data, K, model_x_3.mu, model_x_3.w);
            p_y_x_unlabeled(5, i, y+1) = bmpdf(this_data, K, model_x_4.mu, model_x_4.w);
            p_y_x_unlabeled(6, i, y+1) = bmpdf(this_data, K, model_x_5.mu, model_x_5.w);
            p_y_x_unlabeled(7, i, y+1) = bmpdf(this_data, K, model_x_6.mu, model_x_6.w);
            p_y_x_unlabeled(8, i, y+1) = bmpdf(this_data, K, model_x_7.mu, model_x_7.w);
            p_y_x_unlabeled(9, i, y+1) = bmpdf(this_data, K, model_x_8.mu, model_x_8.w);
            p_y_x_unlabeled(10, i, y+1) = bmpdf(this_data, K, model_x_9.mu, model_x_9.w);  
            
            [~, index] = max(p_y_x_unlabeled(:, i, y+1));
            yhat_x_unlabeled(i, y+1) = index-1;
        else
            p_y_x_unlabeled(:, i, y+1) = 1/10; 
            
            yhat_x_unlabeled(i, y+1) = nan;
        end    
    end
end

% Number of success classified unlabeled data
num_success_unlabeled = zeros(10, 1); 
for y = 0:9
    for i = num_adversarial(y+1)+1:num_unlabeled
        if yhat_x_unlabeled(i, y+1) == y
            num_success_unlabeled(y+1) = num_success_unlabeled(y+1) + 1;
        end
    end
end
percentage_success = num_success_unlabeled./(num_unlabeled - num_adversarial);

num_error_unlabeled_1 = zeros(10, 1); 
for y = 0:9
    for i = num_adversarial(y+1)+1:num_unlabeled
        if (yhat_x_unlabeled(i, y+1) ~= y) 
            num_error_unlabeled_1(y+1) = num_error_unlabeled_1(y+1) + 1;
        end
    end
end
percentage_error_1 = num_error_unlabeled_1./(num_unlabeled - num_adversarial);

num_error_unlabeled_2 = zeros(10, 1); 
for y = 0:9
    for i = 1:num_adversarial(y+1)	
        if (~isnan(yhat_x_unlabeled(i, y+1))) 
            num_error_unlabeled_2(y+1) = num_error_unlabeled_2(y+1) + 1;
        end
    end
end
percentage_error_2 = num_error_unlabeled_2./(num_adversarial);

num_error_unlabeled_3 = zeros(10, 1); 
for y = 0:9
    for i = 1:num_adversarial(y+1)	
        if (~isnan(yhat_x_unlabeled(i, y+1))) && (yhat_x_unlabeled(i, y+1) ~= y)
            num_error_unlabeled_3(y+1) = num_error_unlabeled_3(y+1) + 1;
        end
    end
end
percentage_error_3 = num_error_unlabeled_3./(num_adversarial);

% figure; 
% bar(0:9, [num_labeled*ones(10, 1), num_negative_lrt, num_postivie_lrt, num_testing], 'stacked');
% yyaxis right
% plot(0:9, percentage_success, '-k', 'LineWidth', 3);
% legend({'Labeled', 'Inside Region', 'Outside Region', 'Testing', 'Success classfied unlabeled data %'}, 'Location', 'westoutside');
% ylim([0 1]); 

figure('position', [100, 100, 1000, 600]);
bar(0:9, [num_labeled*ones(10, 1), num_negative_lrt, num_postivie_lrt, num_testing], 'stacked');
legend({'Labeled', 'Inside Similar Region', 'Outside Similar Region', 'Testing'}, 'Location', 'westoutside', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'after_lrt.png'));
saveas(gcf, fullfile(fpath, 'after_lrt.fig'));

%% Refine p(x | y, l = 1) using p(x | y_hat, l = 0)

data_unlabeled_0 = [];
data_unlabeled_1 = [];
data_unlabeled_2 = [];
data_unlabeled_3 = [];
data_unlabeled_4 = [];
data_unlabeled_5 = [];
data_unlabeled_6 = [];
data_unlabeled_7 = [];
data_unlabeled_8 = [];
data_unlabeled_9 = [];
data_unlabeled_new = []; 

for y = 0:9
    for i = 1:num_unlabeled
        switch yhat_x_unlabeled(i, y+1)
            case 0
                data_unlabeled_0 = [data_unlabeled_0, data_unlabeled(:, i, y+1)]; 
            case 1
                data_unlabeled_1 = [data_unlabeled_1, data_unlabeled(:, i, y+1)];
            case 2
                data_unlabeled_2 = [data_unlabeled_2, data_unlabeled(:, i, y+1)];
            case 3
                data_unlabeled_3 = [data_unlabeled_3, data_unlabeled(:, i, y+1)];
            case 4
                data_unlabeled_4 = [data_unlabeled_4, data_unlabeled(:, i, y+1)];
            case 5
                data_unlabeled_5 = [data_unlabeled_5, data_unlabeled(:, i, y+1)];
            case 6
                data_unlabeled_6 = [data_unlabeled_6, data_unlabeled(:, i, y+1)];
            case 7
                data_unlabeled_7 = [data_unlabeled_7, data_unlabeled(:, i, y+1)];
            case 8
                data_unlabeled_8 = [data_unlabeled_8, data_unlabeled(:, i, y+1)];
            case 9
                data_unlabeled_9 = [data_unlabeled_9, data_unlabeled(:, i, y+1)];
            otherwise
                data_unlabeled_new = [data_unlabeled_new, data_unlabeled(:, i, y+1)];
        end
                
    end        
end

[~, model_x_0_new, ~] = mixBernEm([data_labeled(:, :, 1), data_unlabeled_0], K);
[~, model_x_1_new, ~] = mixBernEm([data_labeled(:, :, 2), data_unlabeled_1], K);
[~, model_x_2_new, ~] = mixBernEm([data_labeled(:, :, 3), data_unlabeled_2], K);
[~, model_x_3_new, ~] = mixBernEm([data_labeled(:, :, 4), data_unlabeled_3], K);
[~, model_x_4_new, ~] = mixBernEm([data_labeled(:, :, 5), data_unlabeled_4], K);
[~, model_x_5_new, ~] = mixBernEm([data_labeled(:, :, 6), data_unlabeled_5], K);
[~, model_x_6_new, ~] = mixBernEm([data_labeled(:, :, 7), data_unlabeled_6], K);
[~, model_x_7_new, ~] = mixBernEm([data_labeled(:, :, 8), data_unlabeled_7], K);
[~, model_x_8_new, ~] = mixBernEm([data_labeled(:, :, 9), data_unlabeled_8], K);
[~, model_x_9_new, ~] = mixBernEm([data_labeled(:, :, 10), data_unlabeled_9], K);
[~, model_x_u_new, ~] = mixBernEm(data_unlabeled_new, K);

p_y_0 = (size(data_unlabeled_0, 2)+num_labeled)/(size_training*10);
p_y_1 = (size(data_unlabeled_1, 2)+num_labeled)/(size_training*10);
p_y_2 = (size(data_unlabeled_2, 2)+num_labeled)/(size_training*10);
p_y_3 = (size(data_unlabeled_3, 2)+num_labeled)/(size_training*10);
p_y_4 = (size(data_unlabeled_4, 2)+num_labeled)/(size_training*10);
p_y_5 = (size(data_unlabeled_5, 2)+num_labeled)/(size_training*10);
p_y_6 = (size(data_unlabeled_6, 2)+num_labeled)/(size_training*10);
p_y_7 = (size(data_unlabeled_7, 2)+num_labeled)/(size_training*10);
p_y_8 = (size(data_unlabeled_8, 2)+num_labeled)/(size_training*10);
p_y_9 = (size(data_unlabeled_9, 2)+num_labeled)/(size_training*10);
p_y_u = (size(data_unlabeled_new, 2))/(size_training*10);

% sum([p_y_0 p_y_1 p_y_2 p_y_3 p_y_4 p_y_5 p_y_6 p_y_7 p_y_8 p_y_9 p_y_u])

p_x_new = @(this_data) (p_y_0*bmpdf(this_data, K, model_x_0_new.mu, model_x_0_new.w)+...
    p_y_1*bmpdf(this_data, K, model_x_1_new.mu, model_x_1_new.w)+...
    p_y_2*bmpdf(this_data, K, model_x_2_new.mu, model_x_2_new.w)+...
    p_y_3*bmpdf(this_data, K, model_x_3_new.mu, model_x_3_new.w)+...
    p_y_4*bmpdf(this_data, K, model_x_4_new.mu, model_x_4_new.w)+...
    p_y_5*bmpdf(this_data, K, model_x_5_new.mu, model_x_5_new.w)+...
    p_y_6*bmpdf(this_data, K, model_x_6_new.mu, model_x_6_new.w)+...
    p_y_7*bmpdf(this_data, K, model_x_7_new.mu, model_x_7_new.w)+...
    p_y_8*bmpdf(this_data, K, model_x_8_new.mu, model_x_8_new.w)+...
    p_y_9*bmpdf(this_data, K, model_x_9_new.mu, model_x_9_new.w)+...
    p_y_u*bmpdf(this_data, K, model_x_u_new.mu, model_x_u_new.w));


%% Test

p_y_x_testing = zeros(10, size_testing, 10);
p_y_x_max = zeros(size_testing, 10);
yhat_x_testing = zeros(size_testing, 10);
for y = 0:9
    for i = 1:size_testing
        this_data = data_testing(:, i, y+1);   

        p_y_x_testing(1, i, y+1) = (p_y_0*bmpdf(this_data, K, model_x_0_new.mu, model_x_0_new.w)+.1*p_y_u*bmpdf(this_data, K, model_x_u_new.mu, model_x_u_new.w))/p_x_new(this_data);
        p_y_x_testing(2, i, y+1) = (p_y_1*bmpdf(this_data, K, model_x_1_new.mu, model_x_1_new.w)+.1*p_y_u*bmpdf(this_data, K, model_x_u_new.mu, model_x_u_new.w))/p_x_new(this_data);
        p_y_x_testing(3, i, y+1) = (p_y_2*bmpdf(this_data, K, model_x_2_new.mu, model_x_2_new.w)+.1*p_y_u*bmpdf(this_data, K, model_x_u_new.mu, model_x_u_new.w))/p_x_new(this_data);
        p_y_x_testing(4, i, y+1) = (p_y_3*bmpdf(this_data, K, model_x_3_new.mu, model_x_3_new.w)+.1*p_y_u*bmpdf(this_data, K, model_x_u_new.mu, model_x_u_new.w))/p_x_new(this_data);
        p_y_x_testing(5, i, y+1) = (p_y_4*bmpdf(this_data, K, model_x_4_new.mu, model_x_4_new.w)+.1*p_y_u*bmpdf(this_data, K, model_x_u_new.mu, model_x_u_new.w))/p_x_new(this_data);
        p_y_x_testing(6, i, y+1) = (p_y_5*bmpdf(this_data, K, model_x_5_new.mu, model_x_5_new.w)+.1*p_y_u*bmpdf(this_data, K, model_x_u_new.mu, model_x_u_new.w))/p_x_new(this_data);
        p_y_x_testing(7, i, y+1) = (p_y_6*bmpdf(this_data, K, model_x_6_new.mu, model_x_6_new.w)+.1*p_y_u*bmpdf(this_data, K, model_x_u_new.mu, model_x_u_new.w))/p_x_new(this_data);
        p_y_x_testing(8, i, y+1) = (p_y_7*bmpdf(this_data, K, model_x_7_new.mu, model_x_7_new.w)+.1*p_y_u*bmpdf(this_data, K, model_x_u_new.mu, model_x_u_new.w))/p_x_new(this_data);
        p_y_x_testing(9, i, y+1) = (p_y_8*bmpdf(this_data, K, model_x_8_new.mu, model_x_8_new.w)+.1*p_y_u*bmpdf(this_data, K, model_x_u_new.mu, model_x_u_new.w))/p_x_new(this_data);
        p_y_x_testing(10, i, y+1) = (p_y_9*bmpdf(this_data, K, model_x_9_new.mu, model_x_9_new.w)+.1*p_y_u*bmpdf(this_data, K, model_x_u_new.mu, model_x_u_new.w))/p_x_new(this_data);  

        [~, index] = max(p_y_x_testing(:, i, y+1));
        yhat_x_testing(i, y+1) = index-1;
        
        p_y_x_max(i, y+1) = max(p_y_x_testing(:, i, y+1));
    end
end

% Number of errors 
num_error_testing = zeros(10, 1); 
for y = 0:9
    for i = 1:num_adversarial_testing(1)
        if (p_y_x_max(i, y+1) > .5) && (yhat_x_testing(i, y+1) ~= y)
            num_error_testing(y+1) = num_error_testing(y+1) + 1;
        end
    end
end
percentage_error_testing = num_error_testing/num_adversarial_testing(1);

num_reject_testing = zeros(10, 1); 
for y = 0:9
    for i = 1:num_adversarial_testing(1)
        if (p_y_x_max(i, y+1) <= .5)
            num_reject_testing(y+1) = num_reject_testing(y+1) + 1;
        end
    end
end
percentage_reject_testing = num_reject_testing/num_adversarial_testing(1);

confusion_matrix_robust = eye(10, 10); 
for row = 0:9
    for col = 0:9
        confusion_matrix_robust(row+1, col+1) = sum(yhat_x_testing(num_adversarial_testing+1:end, col+1) == row); 
    end
end
confusion_matrix_robust = confusion_matrix_robust/(size_testing - num_adversarial_testing(1));
sum(confusion_matrix_robust, 1)

figure; 
heatmap({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}, ...
    {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}, confusion_matrix_robust);
saveas(gcf, fullfile(fpath, 'confusion_matrix_robust.png'));
saveas(gcf, fullfile(fpath, 'confusion_matrix_robust.fig'));


% figure('position', [100, 100, 600, 600]);
% hold on;
% bar(0:9, [percentage_error_testing, percentage_success, percentage_error_1, percentage_error_2, percentage_error_3]);
% ylim([0 1])
