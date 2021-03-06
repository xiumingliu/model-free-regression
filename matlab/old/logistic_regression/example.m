clear all
close all

%% Generate data
N = 500; 
N_labeled = 100; 
N_unlabeled = N - N_labeled; 
M = 50;
D = 2;
K = 5;
fpath = 'figures2'; 

% Generate random inputs with a Gaussian mixture model
mu_x = [1 1; -1 -1];
% mu_x = [0 0; 0 0];
COV_x = cat(3, [1 .5; .5 1], [1 -.5; -.5 1]);
p = ones(1, 2)/2;
gm = gmdistribution(mu_x, COV_x, p);

X = random(gm, N+M);
% X_training = X(1:N, :);
X_training = X(1:N_labeled, :);
X_testing = X(N+1:N+M, :);

% Construct the GP: g ~ GP
alpha = 1e6; theta = 1;
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
y = round(logsig(z));
y_training = y(1:N_labeled);
y_testing = y(N+1:N+M);

idx_training = zeros(N_labeled, 1);
for n = 1:N_labeled
    if y_training(n) < .5
        idx_training(n) = 0;
    else
        idx_training(n) = 1;
    end
end

idx_testing = zeros(M, 1);
for m = 1:M
    if y_testing(m) < .5
        idx_testing(m) = 0;
    else
        idx_testing(m) = 1;
    end
end

% Visualize data
figure('position', [100, 100, 600, 600]); % Marginal distribution of y
hist(y_training)
ylim([0 500]);
xlabel('$y$', 'Interpreter', 'latex');
ylabel('Histogram of $y$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'histogram.png'));

%% Training
[~, model_x, ~] = mixGaussVb(X_training', K);
N_k = sum(model_x.R);
mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model_x.R(:, k);
    mu_hat(:, k) = (1/N_k(k))*sum(r_nk.*X_training);
end
COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_x.R(:, k);
    COV_hat(:, :, k) = (1/N_k(k))*((sqrt(r_nk').*(X_training' - mu_hat(:, k)))*(sqrt(r_nk').*(X_training' - mu_hat(:, k)))');
end
pi_hat = (model_x.alpha/sum(model_x.alpha));
gm_x = gmdistribution(mu_hat', COV_hat, pi_hat);


figure('position', [100, 100, 600, 600]); hold on;
scatter(X_training((idx_training==0),1), X_training((idx_training==0),2), 50, 'ok');
scatter(X_training((idx_training==1),1), X_training((idx_training==1),2), 50, 'xk');
% scatter(X_training(N_labeled+1:N, 1), X_training(N_labeled+1:N, 2), [], 's', ...
%     'MarkerFaceColor','r','MarkerEdgeColor','r','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
fcontour(@(x1, x2)pdf(gm_x, [x1 x2]), [-5 5 -5 5], 'LevelList', [.0001 .001 .01 .05:.1:.95 .99 .999 .9999], 'LineWidth', 1)
legend('Labeled, class 1', 'Labeled, class 2', 'Unlabeled');
colorbar;
colormap(jet);
% legend('off');
xlim([-5 5]);
ylim([-5 5]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training.png'));

    

% For p(x | y = 0)
y_0 = y_training(y_training < .5);
X_0 = X_training(y_training < .5, :);
q_y0 = length(y_0)/length(y);
[~, model_y0, ~] = mixGaussVb(X_0', K);
N_k = sum(model_y0.R);
mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model_y0.R(:, k);
    mu_hat(:, k) = (1/N_k(k))*sum(r_nk.*X_0);
end
COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_y0.R(:, k);
    COV_hat(:, :, k) = (1/N_k(k))*((sqrt(r_nk').*(X_0' - mu_hat(:, k)))*(sqrt(r_nk').*(X_0' - mu_hat(:, k)))');
end
pi_hat = (model_y0.alpha/sum(model_y0.alpha));
gm_y0 = gmdistribution(mu_hat', COV_hat, pi_hat);

% For p(x | y = 1)
y_1 = y_training(y_training > .5);
X_1 = X_training(y_training > .5, :);
q_y1 = length(y_1)/length(y);
[~, model_y1, ~] = mixGaussVb(X_1', K);
N_k = sum(model_y1.R);
mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model_y1.R(:, k);
    mu_hat(:, k) = (1/N_k(k))*sum(r_nk.*X_1);
end
COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_y1.R(:, k);
    COV_hat(:, :, k) = (1/N_k(k))*((sqrt(r_nk').*(X_1' - mu_hat(:, k)))*(sqrt(r_nk').*(X_1' - mu_hat(:, k)))');
end
pi_hat = (model_y1.alpha/sum(model_y1.alpha));
gm_y1 = gmdistribution(mu_hat', COV_hat, pi_hat);

% Scaling factor
h = @(x1, x2) (pdf(gm_x, [x1, x2]))/...
    (pdf(gm_y1, [x1, x2])*q_y1 + pdf(gm_y0, [x1, x2])*q_y0);

figure('position', [100, 100, 600, 600]);
hold on
scatter(X_0(:, 1), X_0(:, 2), 50, 'ok'); 
fcontour(@(x1, x2)(h(x1, x2)*pdf(gm_y0, [x1 x2])), [-5 5 -5 5], 'LevelList', [.0001 .001 .01 .05:.1:.95 .99 .999 .9999], 'LineWidth', 1)
colorbar;
colormap(jet);
legend('off');
xlim([-5 5]);
ylim([-5 5]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training_0.png'));

figure('position', [100, 100, 600, 600]);
hold on
scatter(X_1(:, 1), X_1(:, 2), 50, 'xk'); 
fcontour(@(x1, x2)(h(x1, x2)*pdf(gm_y1, [x1 x2])), [-5 5 -5 5], 'LevelList', [.0001 .001 .01 .05:.1:.95 .99 .999 .9999], 'LineWidth', 1)
colorbar;
colormap(jet);
legend('off');
xlim([-5 5]);
ylim([-5 5]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training_1.png'));

%% Testing
post_1 = zeros(M, 1);
post_0 = zeros(M, 1);
y_predict = zeros(M, 1);
w_1 = zeros(M, 1);
w_0 = zeros(M, 1);

for m = 1:M
    this_x_testing = [X_testing(m,1), X_testing(m,2)];
    % w_x = q(x | y)/q(x)
    w_1(m) = h(this_x_testing(1), this_x_testing(2))*pdf(gm_y1, this_x_testing)/pdf(gm_x, this_x_testing);
    w_0(m) = h(this_x_testing(1), this_x_testing(2))*pdf(gm_y0, this_x_testing)/pdf(gm_x, this_x_testing);
    post_1(m) = w_1(m)*q_y1;
    post_0(m) = w_0(m)*q_y0;
    if post_1(m) >= post_0(m)
        y_predict(m) = 1;
    else
        y_predict(m) = 0;
    end
end
num_errors = nnz(y_testing - y_predict);
mse = mean((y_testing - post_1).^2);

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
scatter(X_testing((y_predict==0),1), X_testing((y_predict==0),2), 50, 'ok');
scatter(X_testing((y_predict==1),1), X_testing((y_predict==1),2), 50, 'xk');
scatter(X_testing(find(y_testing - y_predict), 1), X_testing(find(y_testing - y_predict), 2), 100, 'square', 'r', 'filled'); 
fcontour(@(x1, x2)(pdf(gm_y1, [x1, x2])/pdf(gm_x, [x1, x2])*q_y1 - pdf(gm_y0, [x1, x2])/pdf(gm_x, [x1, x2])*q_y0), [-5 5 -5 5], '-b', 'LevelList', [0], 'LineWidth', 1)
xlim([-5 5]);
ylim([-5 5]);
% caxis([-1 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'scatter_testing.png'));

% Error probability
y_predict_EP = zeros(M, 1); 
for m = 1:M
    this_x_testing = [X_testing(m,1), X_testing(m,2)];
    if y_predict(m) == 0
        y_predict_EP(m) = h(this_x_testing(1), this_x_testing(2))...
            *pdf(gm_y1, this_x_testing)./pdf(gm_x, this_x_testing)*q_y1;
    else
        y_predict_EP(m) = h(this_x_testing(1), this_x_testing(2))...
            *pdf(gm_y0, this_x_testing)./pdf(gm_x, this_x_testing)*q_y0;
    end
end

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
scatter(X_testing((y_predict==0),1), X_testing((y_predict==0),2), 50, y_predict_EP((y_predict==0)), 'o', 'LineWidth', 3);
scatter(X_testing((y_predict==1),1), X_testing((y_predict==1),2), 50, y_predict_EP((y_predict==1)), 'x', 'LineWidth', 3);
% scatter(X_testing(find(y_testing - y_predict), 1), X_testing(find(y_testing - y_predict), 2), 100, 'square', 'r', 'filled'); 
xlim([-5 5]);
ylim([-5 5]);
colormap(jet)
colorbar;
% caxis([-1 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'scatter_probability_error.png'));

%% Simulation
figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
scatter(X_testing((y_testing==0),1), X_testing((y_testing==0),2), 50, 'ok');
scatter(X_testing((y_testing==1),1), X_testing((y_testing==1),2), 50, 'xk');

llist = [-.9999 -.999 -.99 -0.95:.5:.95 .99 .999 .9999];
linecolors = {'y', 'm', 'c', 'r', 'g', 'b', 'w', 'k'};
for num_sim = 1:20
% Approximate p(x) with VB-GMM
N_k = sum(model_x.R);

this_mu = (model_x.m)';
this_cov = zeros(D, D, K);
this_W = model_x.W;
for k = 1:K
    this_W(2, 1, k) = this_W(1, 2, k);
    this_cov(:, :, k) = (model_x.beta(k)*this_W(:, :, k))^-1;
    [~, isspd] = chol(this_cov(:, :, k));
    multiplier = 1;
    while isspd
        this_cov(:, :, k) = this_cov(:, :, k) + (0.01*10^(multiplier))*eye(D, D);
        multiplier = multiplier + 1;
        [~, isspd] = chol(this_cov(:, :, k));
    end 
end
mu_hat = mvnrnd(this_mu, this_cov)';

COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_x.R(:, k);
    COV_hat(:, :, k) = (1/N_k(k))*((sqrt(r_nk').*(X_training' - mu_hat(:, k)))*(sqrt(r_nk').*(X_training' - mu_hat(:, k)))');
end
pi_hat = (model_x.alpha/sum(model_x.alpha));

gm_x = gmdistribution(mu_hat', COV_hat, pi_hat);

% For p(x | y = 0)
N_k = sum(model_y0.R);

this_mu = (model_y0.m)';
this_cov = zeros(D, D, K);
this_W = model_y0.W;
for k = 1:K
    this_W(2, 1, k) = this_W(1, 2, k);
    this_cov(:, :, k) = (model_y0.beta(k)*this_W(:, :, k))^-1;
    [~, isspd] = chol(this_cov(:, :, k));
    multiplier = 1;
    while isspd
        this_cov(:, :, k) = this_cov(:, :, k) + (0.01*10^(multiplier))*eye(D, D);
        multiplier = multiplier + 1;
        [~, isspd] = chol(this_cov(:, :, k));
    end 
end
mu_hat = mvnrnd(this_mu, this_cov)';

COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_y0.R(:, k);
    COV_hat(:, :, k) = (1/N_k(k))*((sqrt(r_nk').*(X_0' - mu_hat(:, k)))*(sqrt(r_nk').*(X_0' - mu_hat(:, k)))');
end
pi_hat = (model_y0.alpha/sum(model_y0.alpha));

gm_y0 = gmdistribution(mu_hat', COV_hat, pi_hat);

% For p(x | y = 1)
N_k = sum(model_y1.R);

this_mu = (model_y1.m)';
this_cov = zeros(D, D, K);
this_W = model_y1.W;
for k = 1:K
    this_W(2, 1, k) = this_W(1, 2, k);
    this_cov(:, :, k) = (model_y1.beta(k)*this_W(:, :, k))^-1;
    [~, isspd] = chol(this_cov(:, :, k));
    multiplier = 1;
    while isspd
        this_cov(:, :, k) = this_cov(:, :, k) + (0.01*10^(multiplier))*eye(D, D);
        multiplier = multiplier + 1;
        [~, isspd] = chol(this_cov(:, :, k));
    end 
end
mu_hat = mvnrnd(this_mu, this_cov)';

COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_y1.R(:, k);
    COV_hat(:, :, k) = (1/N_k(k))*((sqrt(r_nk').*(X_1' - mu_hat(:, k)))*(sqrt(r_nk').*(X_1' - mu_hat(:, k)))');
end
pi_hat = (model_y1.alpha/sum(model_y1.alpha));

gm_y1 = gmdistribution(mu_hat', COV_hat, pi_hat);


% Scaling
h = @(x1, x2) (pdf(gm_x, [x1, x2]))/...
    (pdf(gm_y1, [x1, x2])*q_y1 + pdf(gm_y0, [x1, x2])*q_y0);

% Testing
% post_1 = zeros(M, 1);
% post_0 = zeros(M, 1);
% y_predict = zeros(M, 1);
% w_1 = zeros(M, 1);
% w_0 = zeros(M, 1);
% 
% for m = 1:M
%     this_x_testing = [X_testing(m,1), X_testing(m,2)];
%     % w_x = q(x | y)/q(x)
%     w_1(m) = h(this_x_testing(1), this_x_testing(2))*pdf(gm_y1, this_x_testing)/pdf(gm_x, this_x_testing);
%     w_0(m) = h(this_x_testing(1), this_x_testing(2))*pdf(gm_y0, this_x_testing)/pdf(gm_x, this_x_testing);
%     post_1(m) = w_1(m)*q_y1;
%     post_0(m) = w_0(m)*q_y0;
%     if post_1(m) >= post_0(m)
%         y_predict(m) = 1;
%     else
%         y_predict(m) = 0;
%     end
% end
% 
% num_errors = nnz(y_testing - y_predict);
% mse = mean((y_testing - post_1).^2);

% scatter(X_testing(find(y_testing - y_predict), 1), X_testing(find(y_testing - y_predict), 2), 100, 'square', 'r', 'filled'); 
% fcontour(@(x1, x2)(pdf(gm_y1, [x1, x2])/pdf(gm_x, [x1, x2])*q_y1 -...
%     pdf(gm_y0, [x1, x2])/pdf(gm_x, [x1, x2])*q_y0),...
%     [min(min(X_testing)) max(max(X_testing)) min(min(X_testing)) max(max(X_testing))],...
%     '--b', 'LevelList', [0], 'LineWidth', 1);

fcontour(@(x1, x2)(pdf(gm_y1, [x1, x2])./pdf(gm_x, [x1, x2])*q_y1 -...
    pdf(gm_y0, [x1, x2])./pdf(gm_x, [x1, x2])*q_y0),...
    [-5 5 -5 5],...
    '--b', 'LevelList', [0], 'LineWidth', 1);

end

% decision = @(x1, x2)(pdf(gm_y1, [x1, x2])/pdf(gm_x, [x1, x2])*q_y1...
%     - pdf(gm_y0, [x1, x2])/pdf(gm_x, [x1, x2])*q_y0);
% 
% fsolve(decision)

xlim([-5 5]);
ylim([-5 5]);
% caxis([-1 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'simulation.png'));