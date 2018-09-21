clear all, close all

%% Generate data from a non-parametric logistic model
N = 1000;   % Number of training data
N_labeled = 300; 
N_unlabeled = N - N_labeled; 

M = 100;    % Number of testing data
D = 2;      % Dimensions of inputs
K = 20;     % Initial number of categories of GMMs 
fpath = 'figures'; 

% Generate random inputs with a Gaussian mixture model
mu_x = [1.5 1.5; -1.5 -1.5];
COV_x = cat(3, [1 .5; .5 1], [1 -.5; -.5 1]);
p = ones(1, 2)/2;
gm = gmdistribution(mu_x, COV_x, p);

X = random(gm, N+M);
X_training = X(1:N, :);
X_testing = X(N+1:N+M, :);

% Construct the GP: g ~ GP
alpha = 100000; theta = 1;
f_mean = @(x) zeros(length(x), 1);
f_cov = @(x1, x2) alpha^2*exp(-(norm(x1 - x2))^2/(2*theta^2));

mu_g = f_mean(X);
COV_g = zeros(N+M, N+M);
for row = 1:N+M
    for col = 1:N+M
        COV_g(row, col) = f_cov(X(row, :), X(col, :));
    end
end

g = mvnrnd(mu_g, COV_g)';

% z = g + epsilon
var_epsilon = .01; 
epsilon = normrnd(0, sqrt(var_epsilon), N+M, 1);

z = g + epsilon; 

% Generate outputs y
y = logsig(z);
y_training = y(1:N_labeled);
y_testing = y(N+1:N+M);

% Visualize data
figure; % Marginal distribution of y
hist(y_training)
ylim([0 1000]);
xlabel('y');
ylabel('Histogram of output y');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'histogram.png'));

figure; % Scatter plot 
hold on;
scatter(X_training(1:N_labeled, 1), X_training(1:N_labeled, 2), [], y_training, 'filled'); colorbar;
scatter(X_training(N_labeled+1:N, 1), X_training(N_labeled+1:N, 2), [], ...
    'MarkerFaceColor','r','MarkerEdgeColor','r','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('x_1');
ylabel('x_2');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'scatter_training.png'));

figure; % Scatter plot 
hold on;
scatter(X_testing(:, 1), X_testing(:, 2), [], y_testing, 'filled'); colorbar;
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('x_1');
ylabel('x_2');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'scatter_testing.png'));

%% Approximate p(x) with VB-GMM
[~, model, ~] = mixGaussVb(X_training', K);
N_k = sum(model.R);
mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model.R(:, k);
    mu_hat(:, k) = (1/N_k(k))*sum(r_nk.*X_training);
end
COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model.R(:, k);
    COV_hat(:, :, k) = (1/N_k(k))*((sqrt(r_nk').*(X_training' - mu_hat(:, k)))*(sqrt(r_nk').*(X_training' - mu_hat(:, k)))');
end
pi_hat = (model.alpha/sum(model.alpha));

gm_x = gmdistribution(mu_hat', COV_hat, pi_hat);

idx = cluster(gm_x, X);

figure;
hold on
gscatter(X(:,1), X(:,2), idx, 'kkkkk', '', 18*ones(1, K));
fcontour(@(x1, x2)pdf(gm_x, [x1 x2]), [-5 5 -5 5], 'LevelList', [.0001 .001 .01 .05:.1:.95 .99 .999 .9999], 'LineWidth', 1)
colorbar;
legend('off');
xlim([-5 5]);
ylim([-5 5]);
xlabel('x_1');
ylabel('x_2');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'gmm_training.png'));

%% For p(x | y = 0)

y_0 = y_training(y_training < .5);
X_0 = X_training(y_training < .5, :);

q_y0 = length(y_0)/length(y_training);

[~, model, ~] = mixGaussVb(X_0', K);
N_k = sum(model.R);
mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model.R(:, k);
    mu_hat(:, k) = (1/N_k(k))*sum(r_nk.*X_0);
end
COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model.R(:, k);
    COV_hat(:, :, k) = (1/N_k(k))*((sqrt(r_nk').*(X_0' - mu_hat(:, k)))*(sqrt(r_nk').*(X_0' - mu_hat(:, k)))');
end
pi_hat = (model.alpha/sum(model.alpha));

gm_y0 = gmdistribution(mu_hat', COV_hat, pi_hat);

idx = cluster(gm_y0, X_0);

figure;
hold on
gscatter(X_0(:, 1), X_0(:, 2), idx, 'kkkkk', '', 18*ones(1, K)); 
fcontour(@(x1, x2)pdf(gm_y0, [x1 x2]), [-5 5 -5 5], 'LevelList', [.0001 .001 .01 .05:.1:.95 .99 .999 .9999], 'LineWidth', 1)
colorbar;
legend('off');
xlim([-5 5]);
ylim([-5 5]);
xlabel('x_1');
ylabel('x_2');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'gmm_training_0.png'));

%% For p(x | y = 1)

y_1 = y_training(y_training > .5);
X_1 = X_training(y_training > .5, :);

q_y1 = length(y_1)/length(y_training);

[~, model, ~] = mixGaussVb(X_1', K);
N_k = sum(model.R);
mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model.R(:, k);
    mu_hat(:, k) = (1/N_k(k))*sum(r_nk.*X_1);
end
COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model.R(:, k);
    COV_hat(:, :, k) = (1/N_k(k))*((sqrt(r_nk').*(X_1' - mu_hat(:, k)))*(sqrt(r_nk').*(X_1' - mu_hat(:, k)))');
end
pi_hat = (model.alpha/sum(model.alpha));

gm_y1 = gmdistribution(mu_hat', COV_hat, pi_hat);

idx = cluster(gm_y1, X_1);

figure;
hold on
gscatter(X_1(:, 1), X_1(:, 2), idx, 'kkkkk', '', 18*ones(1, K)); 
fcontour(@(x1, x2)pdf(gm_y1, [x1 x2]), [-5 5 -5 5], 'LevelList', [.0001 .001 .01 .05:.1:.95 .99 .999 .9999], 'LineWidth', 1)
colorbar;
legend('off');
xlim([-5 5]);
ylim([-5 5]);
xlabel('x_1');
ylabel('x_2');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'gmm_training_1.png'));

%% Testing
mu_y_test = zeros(M, 1);
y_predict = zeros(M, 1);
w_1 = zeros(M, 1);

for m = 1:M
    % w_x = q(x | y)/q(x)
    w_1(m) = pdf(gm_y1, X_testing(m, :))/pdf(gm_x, X_testing(m, :));
    mu_y_test(m) = w_1(m)*q_y1;
    if mu_y_test(m) >= .5
        y_predict(m) = 1;
    else
        y_predict(m) = 0;
    end
end

num_errors = nnz(y_testing - y_predict);
mse = mean((y_testing - mu_y_test).^2);

f_weight_1 = @(x1, x2) pdf(gm_y1, [x1, x2])/pdf(gm_x, [x1, x2]);
f_weight_0 = @(x1, x2) pdf(gm_y0, [x1, x2])/pdf(gm_x, [x1, x2]);

figure; % Scatter plot 
hold on;
scatter(X_testing(:, 1), X_testing(:, 2), [], mu_y_test, 'filled'); colorbar;
fcontour(@(x1, x2)pdf(gm_y1, [x1, x2])/pdf(gm_x, [x1, x2]), [-5 5 -5 5], 'LevelList', [.0001 .001 .01 .05:.1:.95 .99 .999 .9999], 'LineWidth', 1)
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('x_1');
ylabel('x_2');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'scatter_testing_mu.png'));

figure; % Scatter plot 
hold on;
scatter(X_testing(:, 1), X_testing(:, 2), [], y_predict, 'filled'); colorbar;
scatter(X_testing(find(y_testing - y_predict), 1), X_testing(find(y_testing - y_predict), 2), 150, 'square', 'r', 'filled'); 
fcontour(@(x1, x2)pdf(gm_y1, [x1, x2])/pdf(gm_x, [x1, x2]), [-5 5 -5 5], 'LevelList', [.0001 .001 .01 .05:.1:.95 .99 .999 .9999], 'LineWidth', 1)
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('x_1');
ylabel('x_2');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'scatter_testing_predict.png'));

f_var = @(x1, x2) (0 - f_weight_1(x1, x2)*q_y1)^2*f_weight_0(x1, x2)*q_y0 + (1 - f_weight_1(x1, x2)*q_y1)^2*f_weight_1(x1, x2)*q_y1;
max_var = 0.5*0.5;

figure; % Scatter plot 
hold on;
scatter(X_testing(:, 1), X_testing(:, 2), [], 'k', 'filled'); colorbar;
scatter(X_testing(find(y_testing - y_predict), 1), X_testing(find(y_testing - y_predict), 2), 150, 'square', 'r', 'filled'); 
fcontour(f_var, [-5 5 -5 5], 'LevelList', max_var*[.0001 .001 .01 .05:.1:.95 .99 .999 .9999], 'LineWidth', 1)
xlim([-5 5]);
ylim([-5 5]);
caxis([0 max_var]);
xlabel('x_1');
ylabel('x_2');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'scatter_testing_variance.png'));