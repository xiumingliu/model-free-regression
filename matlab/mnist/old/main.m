clear all
close all
fclose all 

%% All data 0-9
data = zeros(28, 28, 1000, 10);
size_data = 100;
for n = 1:size_data
    filename = ['data' num2str(n-1)];
    fid=fopen(filename, 'r'); % open the file corresponding to digit 8
    for i = 1:size_data
    data(:, :, i, n) = fread(fid, [28 28], 'uchar');
    end
end

%% Training data and testing data
size_training = .9*size_data; 
size_testing = .1*size_data; 

data_training = data(:, :, 1:size_training, :);
data_testing = data(:, :, size_training+1:end, :);

%% Labeled and unlabeled training data
% Labeled, num_labeled from 1 to 900

data_labeled_y = cell(10, 1);
num_labeled = zeros(10, 1);
for y = 0:9
    num_labeled(y+1) = randi(size_training); 
    data_labeled_y(y+1) = {data_training(:, :, 1:num_labeled(y+1), y+1)};
end

data_labeled = zeros(28, 28, sum(num_labeled)); 
for y = 0:9
    data_labeled(:, :, 1+sum(num_labeled(1:y)):sum(num_labeled(1:y+1)))...
        = data_training(:, :, 1:num_labeled(y+1), y+1);
end

% Unlabeled, num_unlabeled = 900 - num_labeled
num_unlabeled = size_training*ones(10, 1) - num_labeled;
data_unlabeled = zeros(28, 28, sum(num_unlabeled)); 
for y = 0:9
    data_unlabeled(:, :, 1+sum(num_unlabeled(1:y)):sum(num_unlabeled(1:y+1)))...
        = data_training(:, :, num_labeled(y+1)+1:end, y+1);
end

figure; 
bar(0:9, [num_labeled, num_unlabeled], 'stacked');

%% p(x | y, l = 1)
for y = 0:9
    this_data = data_labeled_y{y+1};
    
    p_x_y_labled(y+1).mu = reshape(sum(this_data, 3)/num_labeled(y+1), 28^2, 1);
    
    this_mu = p_x_y_labled(y+1).mu; 
    p_x_y_labled(y+1).sigma = zeros(28^2, 28^2);
    for i = 1:num_labeled(y+1)
        this_data_vector = reshape(this_data(:, :, i), 28^2, 1); 
        p_x_y_labled(y+1).sigma = p_x_y_labled(y+1).sigma + ...
            (this_data_vector - this_mu_vector)*(this_data_vector - this_mu_vector)';
    end
    p_x_y_labled(y+1).sigma = p_x_y_labled(y+1).sigma/num_labeled(y+1);
    p_x_y_labled(y+1).sigma = p_x_y_labled(y+1).sigma + 1e-6*eye(28^2, 28^2);   % Assure PSD
end

this_data = reshape(data_labeled(:, :, 1), 28^2, 1); 
(1/sqrt((2*pi)^(28^2))*det(p_x_y_labled(1).sigma))*exp(-.5*(this_data - p_x_y_labled(1).mu)'*inv(p_x_y_labled(1).sigma)*(this_data - p_x_y_labled(1).mu))

figure; 
subplot(2, 5, 1);
imagesc(reshape(p_x_y_labled(1).mu, 28, 28)');
subplot(2, 5, 2);
imagesc(reshape(p_x_y_labled(2).mu, 28, 28)');
subplot(2, 5, 3);
imagesc(reshape(p_x_y_labled(3).mu, 28, 28)');
subplot(2, 5, 4);
imagesc(reshape(p_x_y_labled(4).mu, 28, 28)');
subplot(2, 5, 5);
imagesc(reshape(p_x_y_labled(5).mu, 28, 28)');
subplot(2, 5, 6);
imagesc(reshape(p_x_y_labled(6).mu, 28, 28)');
subplot(2, 5, 7);
imagesc(reshape(p_x_y_labled(7).mu, 28, 28)');
subplot(2, 5, 8);
imagesc(reshape(p_x_y_labled(8).mu, 28, 28)');
subplot(2, 5, 9);
imagesc(reshape(p_x_y_labled(9).mu, 28, 28)');
subplot(2, 5, 10);
imagesc(reshape(p_x_y_labled(10).mu, 28, 28)');

figure; 
subplot(2, 5, 1);
imagesc(p_x_y_labled(1).sigma);
subplot(2, 5, 2);
imagesc(p_x_y_labled(2).sigma);
subplot(2, 5, 3);
imagesc(p_x_y_labled(3).sigma);
subplot(2, 5, 4);
imagesc(p_x_y_labled(4).sigma);
subplot(2, 5, 5);
imagesc(p_x_y_labled(5).sigma);
subplot(2, 5, 6);
imagesc(p_x_y_labled(6).sigma);
subplot(2, 5, 7);
imagesc(p_x_y_labled(7).sigma);
subplot(2, 5, 8);
imagesc(p_x_y_labled(8).sigma);
subplot(2, 5, 9);
imagesc(p_x_y_labled(9).sigma);
subplot(2, 5, 10);
imagesc(p_x_y_labled(10).sigma);

%% p(x | l = 1)
p_x_labled.mu = reshape(sum(data_unlabeled, 3)/sum(num_unlabeled), 28^2, 1);
p_x_labled.sigma = zeros(28^2, 28^2);
for i = 1:sum(num_unlabeled)
    this_data = reshape(data_unlabeled(:, :, i), 28^2, 1);
    this_mu = reshape(p_x_labled.mu, 28^2, 1); 
 
    p_x_labled.sigma = p_x_labled.sigma + ...
        (this_data - this_mu)*(this_data - this_mu)';
end
p_x_labled.sigma = p_x_labled.sigma/sum(num_unlabeled);
p_x_labled.sigma = p_x_labled.sigma + 1e-6*eye(28^2, 28^2); % Assure PSD
% p_x_labled.sigma = diag(diag(p_x_labled.sigma));

figure;
imagesc(reshape(p_x_labled.mu, 28, 28)');
figure; 
imagesc(p_x_labled.sigma);


%% p(x | l = 0)
p_x_unlabled.mu = reshape(sum(data_unlabeled, 3)/sum(num_unlabeled), 28^2, 1);
p_x_unlabled.sigma = zeros(28^2, 28^2);
for i = 1:sum(num_unlabeled)
    this_data = reshape(data_unlabeled(:, :, i), 28^2, 1);
    this_mu = reshape(p_x_unlabled.mu, 28^2, 1); 
 
    p_x_unlabled.sigma = p_x_unlabled.sigma + ...
        (this_data - this_mu)*(this_data - this_mu)';
end
p_x_unlabled.sigma = p_x_unlabled.sigma/sum(num_unlabeled);
p_x_unlabled.sigma = p_x_unlabled.sigma + 1e-6*eye(28^2, 28^2); % Assure PSD
% p_x_unlabled.sigma = diag(diag(p_x_unlabled.sigma));

figure;
imagesc(reshape(p_x_unlabled.mu, 28, 28)');
figure; 
imagesc(p_x_unlabled.sigma);

%% LRT
log_likelihood_ratio = zeros(sum(num_unlabeled), 1); 
for i = 1:sum(num_unlabeled)
    this_x = reshape(data_unlabeled(:, :, i), 28^2, 1);
    log_likelihood_ratio(i) = (mvnpdf(this_x, p_x_unlabled.mu, p_x_unlabled.sigma)/...
        mvnpdf(this_x, p_x_labled.mu, p_x_labled.sigma));
end






