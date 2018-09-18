clear all
close all

%% Setup 
fpath = 'figures1'; 

N_labeled = 1000;    % Number of labeled training data 
N_unlabeled = 0;  % Number of unlabeld training data
N_testing = 50;     % Number of testing data

D = 2;      % Dimension of the input X
K = 10;     % Number of components used in GMM

level_list = [0.001 .01 .1:.1:.9];

%% Generate synthetic data
run data_generate.m

%% Histogram of outputs
figure('position', [100, 100, 600, 600]); % Marginal distribution of y
hist(y_labeled)
ylim([0 500]);
xlabel('$y$', 'Interpreter', 'latex');
ylabel('Histogram of $y$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'histogram.png'));

%% The marginal distribution p(x), using both labeled and unlabeld data
[~, model_x, ~] = mixGaussVb(X(1:N_labeled+N_unlabeled, :)', K);
this_Nk = sum(model_x.R);
this_mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model_x.R(:, k);
    this_mu_hat(:, k) = (1/this_Nk(k))*sum(r_nk.*X(1:N_labeled+N_unlabeled, :));
end
this_COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_x.R(:, k);
    this_COV_hat(:, :, k) = (1/this_Nk(k))*((sqrt(r_nk').*(X(1:N_labeled+N_unlabeled, :)' - this_mu_hat(:, k)))...
        *(sqrt(r_nk').*(X(1:N_labeled+N_unlabeled, :)' - this_mu_hat(:, k)))');
end
this_pi_hat = (model_x.alpha/sum(model_x.alpha));
p_x = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

% Visualize p(x)
figure('position', [100, 100, 600, 600]); hold on;
scatter(X_labeled((y_labeled==0),1), X_labeled((y_labeled==0),2), 50, 'ok');
scatter(X_labeled((y_labeled==1),1), X_labeled((y_labeled==1),2), 50, 'xk');
scatter(X_unlabeled(:, 1), X_unlabeled(:, 2), [], 's', ...
    'MarkerFaceColor','r','MarkerEdgeColor','r','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
fcontour(@(x1, x2)pdf(p_x, [x1 x2]), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Labeled, class 0', 'Labeled, class 1', 'Unlabeled');
colorbar;
colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training.png'));

%% The conditional distribution p(x | y = 0) and p(x | y = 1), using labeled data
% p(x | y = 0)
y_0 = y_labeled(y_labeled == 0);
X_0 = X_labeled(y_labeled == 0, :);
p_y0 = length(y_0)/length(y);
[~, model_xy0, ~] = mixGaussVb(X_0', K);
this_Nk = sum(model_xy0.R);
this_mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model_xy0.R(:, k);
    this_mu_hat(:, k) = (1/this_Nk(k))*sum(r_nk.*X_0);
end
this_COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_xy0.R(:, k);
    this_COV_hat(:, :, k) = (1/this_Nk(k))*((sqrt(r_nk').*(X_0' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X_0' - this_mu_hat(:, k)))');
end
this_pi_hat = (model_xy0.alpha/sum(model_xy0.alpha));
p_xy0 = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

% p(x | y = 1)
y_1 = y_labeled(y_labeled == 1);
X_1 = X_labeled(y_labeled == 1, :);
p_y1 = length(y_1)/length(y);
[~, model_xy1, ~] = mixGaussVb(X_1', K);
this_N_k = sum(model_xy1.R);
this_mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model_xy1.R(:, k);
    this_mu_hat(:, k) = (1/this_N_k(k))*sum(r_nk.*X_1);
end
this_COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_xy1.R(:, k);
    this_COV_hat(:, :, k) = (1/this_N_k(k))*((sqrt(r_nk').*(X_1' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X_1' - this_mu_hat(:, k)))');
end
this_pi_hat = (model_xy1.alpha/sum(model_xy1.alpha));
p_xy1 = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

% Scaling factor
h = @(x1, x2) (pdf(p_x, [x1, x2]))/(pdf(p_xy1, [x1, x2])*p_y1 + pdf(p_xy0, [x1, x2])*p_y0);

figure('position', [100, 100, 600, 600]);
hold on
scatter(X_0(:, 1), X_0(:, 2), 50, 'ok'); 
fcontour(@(x1, x2)(h(x1, x2)*pdf(p_xy0, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Labeled, class 0');
colorbar;
colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training_0.png'));

figure('position', [100, 100, 600, 600]);
hold on
scatter(X_1(:, 1), X_1(:, 2), 50, 'xk'); 
fcontour(@(x1, x2)(h(x1, x2)*pdf(p_xy1, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Labeled, class 1');
colorbar;
colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training_1.png'));

%% Testing
y_predict = zeros(N_testing, 1);

for n = 1:N_testing
    this_x_testing = [X_testing(n,1), X_testing(n,2)];
    w_y1 = h(this_x_testing(1), this_x_testing(2))*pdf(p_xy1, this_x_testing)/pdf(p_x, this_x_testing);
    w_y0 = h(this_x_testing(1), this_x_testing(2))*pdf(p_xy0, this_x_testing)/pdf(p_x, this_x_testing);
    if w_y1*p_y1 >= w_y0*p_y0
        y_predict(n) = 1;
    else
        y_predict(n) = 0;
    end
end

figure('position', [100, 100, 600, 600]); 
hold on;
scatter(X_testing((y_predict==0),1), X_testing((y_predict==0),2), 50, 'ok');
scatter(X_testing((y_predict==1),1), X_testing((y_predict==1),2), 50, 'xk');
scatter(X_testing(find(y_testing - y_predict), 1), X_testing(find(y_testing - y_predict), 2), 100, 'square', 'r', 'filled'); 
fcontour(@(x1, x2)(pdf(p_xy1, [x1, x2])/pdf(p_x, [x1, x2])*p_y1 - pdf(p_xy0, [x1, x2])/pdf(p_x, [x1, x2])*p_y0), [-5 5 -5 5], '-b', 'LevelList', [0], 'LineWidth', 3)
legend('Correctly labeled 0', 'Correctly labeled 1', 'Errors', 'Decision boundaries');
xlim([-5 5]);
ylim([-5 5]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'scatter_testing.png'));

% Error probability
y_predict_EP = zeros(N_testing, 1); 
for n = 1:N_testing
    this_x_testing = [X_testing(n,1), X_testing(n,2)];
    if y_predict(n) == 0
        y_predict_EP(n) = h(this_x_testing(1), this_x_testing(2))...
            *pdf(p_xy1, this_x_testing)./pdf(p_x, this_x_testing)*p_y1;
    else
        y_predict_EP(n) = h(this_x_testing(1), this_x_testing(2))...
            *pdf(p_xy0, this_x_testing)./pdf(p_x, this_x_testing)*p_y0;
    end
end

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
scatter(X_testing((y_predict==0),1), X_testing((y_predict==0),2), 50, y_predict_EP((y_predict==0)), 'o', 'LineWidth', 3);
scatter(X_testing((y_predict==1),1), X_testing((y_predict==1),2), 50, y_predict_EP((y_predict==1)), 'x', 'LineWidth', 3);
xlim([-5 5]);
ylim([-5 5]);
colormap(jet)
colorbar;
caxis([0 .5]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'scatter_probability_error.png'));

%% Monte Carlo Simulations
run simulations.m
