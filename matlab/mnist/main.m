clear all
close all
fclose all 

%% All data 0-9
data = zeros(28^2, 1000, 10);
size_data = 1000;
for y = 0:9
    filename = ['data' num2str(y)];
    fid=fopen(filename, 'r'); % open the file corresponding to digit 8
    for i = 1:size_data
    data(:, i, y+1) = round(reshape(fread(fid, [28 28], 'uchar'), 28^2, 1)/255);
    end
end

%% Training data and testing data
size_training = .7*size_data; 
size_testing = .3*size_data; 

data_training = data(:, 1:size_training, :);
data_testing = data(:, size_training+1:size_training+size_testing, :);

%% Labeled and unlabeled training data
% Labeled
num_labeled = .3*size_training;
data_labeled = zeros(28^2, num_labeled, 10);
for y = 0:9
    data_labeled(:, :, y+1) = data_training(:, 1:num_labeled, y+1);
end

% Unlabeled, num_unlabeled = 900 - num_labeled
num_unlabeled = size_training - num_labeled;
data_unlabeled = zeros(28^2, num_unlabeled, 10);
for y = 0:9
    data_unlabeled(:, :, y+1) = data_training(:, num_labeled+1:num_labeled+num_unlabeled, y+1);
end

% Adversarial data in the unlabeled
num_adversarial = zeros(10, 1);
for y = 0:9
    num_adversarial(y+1) = randi(0.5*num_unlabeled);
    data_unlabeled(:, 1:num_adversarial(y+1), y+1) = flip(data_unlabeled(:, 1:num_adversarial(y+1), y+1));
end

% Number of testing data
num_testing = size_testing*ones(10, 1);

figure; 
bar(0:9, [num_labeled*ones(10, 1), num_unlabeled*ones(10, 1)-num_adversarial, num_adversarial, num_testing], 'stacked');
legend({'Labeled', 'Unlabeled:good', 'Unlabeled:adversarial', 'Testing'}, 'Location', 'westoutside');

%% Parameters
kappa = 3;
K = 1;

%% p(x | l = 0)
 
[~, model_x_u, ~] = mixBernEm(reshape(data_unlabeled, 28^2, num_unlabeled*10), K);

%% p(x | l = 1)
% K = 10;
[~, model_x_l, ~] = mixBernEm(reshape(data_labeled, 28^2, num_labeled*10), K);

%% p(x | y, l = 1)

[~, model_x_0, ~] = mixBernEm(reshape(data_labeled(:, :, 1), 28^2, num_labeled), K);
[~, model_x_1, ~] = mixBernEm(reshape(data_labeled(:, :, 2), 28^2, num_labeled), K);
[~, model_x_2, ~] = mixBernEm(reshape(data_labeled(:, :, 3), 28^2, num_labeled), K);
[~, model_x_3, ~] = mixBernEm(reshape(data_labeled(:, :, 4), 28^2, num_labeled), K);
[~, model_x_4, ~] = mixBernEm(reshape(data_labeled(:, :, 5), 28^2, num_labeled), K);
[~, model_x_5, ~] = mixBernEm(reshape(data_labeled(:, :, 6), 28^2, num_labeled), K);
[~, model_x_6, ~] = mixBernEm(reshape(data_labeled(:, :, 7), 28^2, num_labeled), K);
[~, model_x_7, ~] = mixBernEm(reshape(data_labeled(:, :, 8), 28^2, num_labeled), K);
[~, model_x_8, ~] = mixBernEm(reshape(data_labeled(:, :, 9), 28^2, num_labeled), K);
[~, model_x_9, ~] = mixBernEm(reshape(data_labeled(:, :, 10), 28^2, num_labeled), K);

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

p_y_x = zeros(10, num_unlabeled, 10);
p_y_x_max = zeros(num_unlabeled, 10);
yhat_x = zeros(num_unlabeled, 10);
for y = 0:9
    for i = 1:num_unlabeled
        this_data = data_unlabeled(:, i, y+1);   
        if LLR_unlabeled(i, y+1) < kappa
            p_y_x(1, i, y+1) = bmpdf(this_data, K, model_x_0.mu, model_x_0.w);
            p_y_x(2, i, y+1) = bmpdf(this_data, K, model_x_1.mu, model_x_1.w);
            p_y_x(3, i, y+1) = bmpdf(this_data, K, model_x_2.mu, model_x_2.w);
            p_y_x(4, i, y+1) = bmpdf(this_data, K, model_x_3.mu, model_x_3.w);
            p_y_x(5, i, y+1) = bmpdf(this_data, K, model_x_4.mu, model_x_4.w);
            p_y_x(6, i, y+1) = bmpdf(this_data, K, model_x_5.mu, model_x_5.w);
            p_y_x(7, i, y+1) = bmpdf(this_data, K, model_x_6.mu, model_x_6.w);
            p_y_x(8, i, y+1) = bmpdf(this_data, K, model_x_7.mu, model_x_7.w);
            p_y_x(9, i, y+1) = bmpdf(this_data, K, model_x_8.mu, model_x_8.w);
            p_y_x(10, i, y+1) = bmpdf(this_data, K, model_x_9.mu, model_x_9.w);  
            
            [~, index] = max(p_y_x(:, i, y+1));
            yhat_x(i, y+1) = index-1;
        else
            p_y_x(:, i, y+1) = 1/10; 
            
            yhat_x(i, y+1) = nan;
        end    
    end
end

% Number of success 
num_success = zeros(10, 1); 
for y = 0:9
    for i = 1:num_unlabeled
        if yhat_x(i, y+1) == y
            num_success(y+1) = num_success(y+1) + 1;
        end
    end
end
percentage_success = num_success./(num_unlabeled - num_adversarial);

figure; 
bar(0:9, [num_labeled*ones(10, 1), num_negative_lrt, num_postivie_lrt, num_testing], 'stacked');
yyaxis right
plot(0:9, percentage_success, '-k', 'LineWidth', 3);
legend({'Labeled', 'Inside Region', 'Outside Region', 'Testing', 'Success classfied unlabeled data %'}, 'Location', 'westoutside');
ylim([0 1]); 

%% Refine p(x | y, l = 1) using p(x | y_hat, l = 0)







