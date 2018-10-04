clear all
close all
fclose all 

fpath = 'figures1'; 

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
size_training = .8*size_data; 
size_testing = .2*size_data; 

data_training = data(:, 1:size_training, :);
data_testing = data(:, size_training+1:size_training+size_testing, :);

%% Labeled and unlabeled training data
% Labeled
num_labeled = .2*size_training;
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
    num_adversarial(y+1) = randi(0.8*num_unlabeled);
    data_unlabeled(:, 1:num_adversarial(y+1), y+1) = flip(data_unlabeled(:, 1:num_adversarial(y+1), y+1));
end

% Adversarial data in the testing
num_adversarial_testing = zeros(10, 1);
for y = 0:9
    num_adversarial_testing(y+1) = .2*size_testing;
    data_testing(:, 1:num_adversarial_testing(y+1), y+1) = flip(data_testing(:, 1:num_adversarial_testing(y+1), y+1));
end

% Number of testing data
num_testing = size_testing*ones(10, 1);

figure('position', [100, 100, 1000, 600]);
bar(0:9, [num_labeled*ones(10, 1), num_unlabeled*ones(10, 1)-num_adversarial, num_adversarial, num_testing- num_adversarial_testing, num_adversarial_testing], 'stacked');
legend({'Labeled', 'Unlabeled: valid', 'Unlabeled: adversarial', 'Testing: valid', 'Testing: adversarial'}, 'Location', 'westoutside', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'data.png'));
saveas(gcf, fullfile(fpath, 'data.fig'));

%% Parameters
kappa = 0;
K = 1;

%% p(x)
[~, model_x, ~] = mixBernEm(reshape(data_training, 28^2, size_training*10), K);

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