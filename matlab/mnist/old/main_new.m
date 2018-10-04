clear all
close all
fclose all 

%% All data 0-9
data = zeros(28^2, 1000, 10);
size_data = 100;
for n = 1:10
    filename = ['data' num2str(n-1)];
    fid=fopen(filename, 'r'); % open the file corresponding to digit 8
    for i = 1:size_data
    data(:, i, n) = reshape(fread(fid, [28 28], 'uchar'), 28^2, 1);
    end
end

%% Training data and testing data
size_training = .9*size_data; 
size_testing = .1*size_data; 

data_training = data(:, 1:size_training, :);
data_testing = data(:, size_training+1:size_training+size_testing, :);

%% Labeled and unlabeled training data
% Labeled, num_labeled from 1 to 900

data_labeled_y = cell(10, 1);
num_labeled = zeros(10, 1);
for y = 0:9
    num_labeled(y+1) = randi(size_training); 
    data_labeled_y(y+1) = {data_training(:, 1:num_labeled(y+1), y+1)};
end

data_labeled = zeros(28^2, sum(num_labeled)); 
for y = 0:9
    data_labeled(:, 1+sum(num_labeled(1:y)):sum(num_labeled(1:y+1)))...
        = data_training(:, 1:num_labeled(y+1), y+1);
end

% Unlabeled, num_unlabeled = 900 - num_labeled
num_unlabeled = size_training*ones(10, 1) - num_labeled;
data_unlabeled = zeros(28^2, sum(num_unlabeled)); 
for y = 0:9
    data_unlabeled(:, 1+sum(num_unlabeled(1:y)):sum(num_unlabeled(1:y+1)))...
        = data_training(:, num_labeled(y+1)+1:end, y+1);
end

figure; 
bar(0:9, [num_labeled, num_unlabeled], 'stacked');

%% p(x | y, l = 1)
for y = 0:9
    this_data = data_labeled_y{y+1};
    
    p_x_y_labled(y+1).mu = sum(this_data, 2)/num_labeled(y+1);
     
    p_x_y_labled(y+1).sigma = zeros(28^2, 28^2);
    for i = 1:num_labeled(y+1)
        this_data_i = this_data(:, i); 
        p_x_y_labled(y+1).sigma = p_x_y_labled(y+1).sigma + ...
            (this_data_i - p_x_y_labled(y+1).mu)*(this_data_i - p_x_y_labled(y+1).mu)';
    end
    p_x_y_labled(y+1).sigma = p_x_y_labled(y+1).sigma/num_labeled(y+1);
    p_x_y_labled(y+1).sigma = p_x_y_labled(y+1).sigma + 1e-6*eye(28^2, 28^2);   % Assure PSD
end

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
p_x_labled.mu = sum(data_labeled, 2)/sum(num_labeled);
p_x_labled.sigma = zeros(28^2, 28^2);
for i = 1:sum(num_labeled)
    this_data_i = data_labeled(:, i);
 
    p_x_labled.sigma = p_x_labled.sigma + ...
        (this_data_i - p_x_labled.mu)*(this_data_i - p_x_labled.mu)';
end
p_x_labled.sigma = p_x_labled.sigma/sum(num_labeled);
p_x_labled.sigma = p_x_labled.sigma + 1e-3*eye(28^2, 28^2); % Assure PSD
p_x_labled.sigma = diag(diag(p_x_labled.sigma));

figure;
imagesc(reshape(p_x_labled.mu, 28, 28)');
figure; 
imagesc(p_x_labled.sigma);


%% p(x | l = 0)
p_x_unlabled.mu = sum(data_unlabeled, 2)/sum(num_unlabeled);
p_x_unlabled.sigma = zeros(28^2, 28^2);
for i = 1:sum(num_unlabeled)
    this_data_i = data_unlabeled(:, i);
 
    p_x_unlabled.sigma = p_x_unlabled.sigma + ...
        (this_data_i - p_x_unlabled.mu)*(this_data_i - p_x_unlabled.mu)';
end
p_x_unlabled.sigma = p_x_unlabled.sigma/sum(num_unlabeled);
p_x_unlabled.sigma = p_x_unlabled.sigma + 1e-3*eye(28^2, 28^2); % Assure PSD
p_x_unlabled.sigma = diag(diag(p_x_unlabled.sigma));

figure;
imagesc(reshape(p_x_unlabled.mu, 28, 28)');
figure; 
imagesc(p_x_unlabled.sigma);

%% LRT

% log_likelihood_ratio_labeled = zeros(sum(num_labeled), 1); 
% for i = 1:sum(num_labeled)
%     this_data_i = data_labeled(:, i);
%     log_likelihood_ratio_labeled(i) = (mvnpdf(this_data_i, p_x_unlabled.mu, p_x_unlabled.sigma)/...
%         mvnpdf(this_data_i, p_x_labled.mu, p_x_labled.sigma));
% end

% log_likelihood_ratio_unlabeled = zeros(sum(num_unlabeled), 1); 
% for i = 1:sum(num_unlabeled)
%     this_data_i = data_unlabeled(:, i);
%     log_likelihood_ratio_unlabeled(i) = (mvnpdf(this_data_i, p_x_unlabled.mu, p_x_unlabled.sigma)/...
%         mvnpdf(this_data_i, p_x_labled.mu, p_x_labled.sigma));
% end

log_likelihood_ratio_test = zeros(size_testing, 10); 
for n = 1:10
for i = 1:size_testing
    this_data_i = data_testing(:, i, n);
    
    pdf_y0 = mvnpdf(this_data_i, p_x_y_labled(1).mu, p_x_y_labled(1).sigma);
    pdf_y1 = mvnpdf(this_data_i, p_x_y_labled(2).mu, p_x_y_labled(2).sigma);
    pdf_y2 = mvnpdf(this_data_i, p_x_y_labled(3).mu, p_x_y_labled(3).sigma);
    pdf_y3 = mvnpdf(this_data_i, p_x_y_labled(4).mu, p_x_y_labled(4).sigma);
    pdf_y4 = mvnpdf(this_data_i, p_x_y_labled(5).mu, p_x_y_labled(5).sigma);
    pdf_y5 = mvnpdf(this_data_i, p_x_y_labled(6).mu, p_x_y_labled(6).sigma);
    pdf_y6 = mvnpdf(this_data_i, p_x_y_labled(7).mu, p_x_y_labled(7).sigma);
    pdf_y7 = mvnpdf(this_data_i, p_x_y_labled(8).mu, p_x_y_labled(8).sigma);
    pdf_y8 = mvnpdf(this_data_i, p_x_y_labled(9).mu, p_x_y_labled(9).sigma);
    pdf_y9 = mvnpdf(this_data_i, p_x_y_labled(10).mu, p_x_y_labled(10).sigma);
    
    log_likelihood_ratio_test(i, n) = (mvnpdf(this_data_i, p_x_unlabled.mu, p_x_unlabled.sigma)/...
        mvnpdf(this_data_i, p_x_labled.mu, p_x_labled.sigma));
end
end







