clear all
close all

%% Setup 
fpath = 'figures5'; 

N_labeled = 50;    % Number of labeled training data 
N_unlabeled = 500;  % Number of unlabeld training data
N_testing = 50;     % Number of testing data

NUM_SIM_DB = 10;
NUM_SIM_EP = 5000;

D = 2;      % Dimension of the input X
K = 50;     % Number of components used in GMM

level_list = [0.001 .01 .1:.1:.9];

%% Generate synthetic data
run data_generate_new.m

%% Histogram of outputs
C = categorical([y_labeled; -1*ones(N_unlabeled, 1)], [0 1 -1],...
    {'Class 0', 'Class 1', 'Unlabeled'});

figure('position', [100, 100, 600, 600]); % Marginal distribution of y
histogram(C)
ylim([0 600]);
xlabel('$y$', 'Interpreter', 'latex');
ylabel('Histogram of $y$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'histogram.png'));
saveas(gcf, fullfile(fpath, 'histogram.fig'));

%% The conditional distribution of unlabeled data 
% y_0 = y_labeled(y_labeled == 0);
% X_0 = X_labeled(y_labeled == 0, :);
p_y_unlabeled = length(y_unlabeled)/(length(y_labeled) + length(y_unlabeled));
[~, model_xy_unlabled, ~] = mixGaussVb(X_unlabeled', K);
this_Nk = sum(model_xy_unlabled.R);
this_mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model_xy_unlabled.R(:, k);
    this_mu_hat(:, k) = (1/this_Nk(k))*sum(r_nk.*X_unlabeled);
end
this_COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_xy_unlabled.R(:, k);
    this_COV_hat(:, :, k) = (1/this_Nk(k))*((sqrt(r_nk').*(X_unlabeled' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X_unlabeled' - this_mu_hat(:, k)))');
end
this_pi_hat = (model_xy_unlabled.alpha/sum(model_xy_unlabled.alpha));
p_xy_unlabeled = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

figure('position', [100, 100, 600, 600]);
hold on
scatter(X_unlabeled(:, 1), X_unlabeled(:, 2), 50, 's', ...
    'MarkerFaceColor','w','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',1);
fcontour(@(x1, x2)(pdf(p_xy_unlabeled, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Unlabeled');
% colorbar;
colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training_unlabeled.png'));
saveas(gcf, fullfile(fpath, 'gmm_training_unlabeled.fig'));

%% The conditional distribution p(x | y = 0) and p(x | y = 1), using labeled data
% p(x | y = 0)
y_0 = y_labeled(y_labeled == 0);
X_0 = X_labeled(y_labeled == 0, :);
p_y_0 = length(y_0)/(length(y_labeled) + length(y_unlabeled));
[~, model_xy_0, ~] = mixGaussVb(X_0', K);
this_Nk = sum(model_xy_0.R);
this_mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model_xy_0.R(:, k);
    this_mu_hat(:, k) = (1/this_Nk(k))*sum(r_nk.*X_0);
end
this_COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_xy_0.R(:, k);
    this_COV_hat(:, :, k) = (1/this_Nk(k))*((sqrt(r_nk').*(X_0' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X_0' - this_mu_hat(:, k)))');
end
this_pi_hat = (model_xy_0.alpha/sum(model_xy_0.alpha));
p_xy_0 = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

% p(x | y = 1)
y_1 = y_labeled(y_labeled == 1);
X_1 = X_labeled(y_labeled == 1, :);
p_y_1 = length(y_1)/length(y);
[~, model_xy_1, ~] = mixGaussVb(X_1', K);
this_N_k = sum(model_xy_1.R);
this_mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model_xy_1.R(:, k);
    this_mu_hat(:, k) = (1/this_N_k(k))*sum(r_nk.*X_1);
end
this_COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_xy_1.R(:, k);
    this_COV_hat(:, :, k) = (1/this_N_k(k))*((sqrt(r_nk').*(X_1' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X_1' - this_mu_hat(:, k)))');
end
this_pi_hat = (model_xy_1.alpha/sum(model_xy_1.alpha));
p_xy_1 = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

figure('position', [100, 100, 600, 600]);
hold on
scatter(X_0(:, 1), X_0(:, 2), 50, 'ok'); 
fcontour(@(x1, x2)(pdf(p_xy_0, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Labeled, class 0');
% colorbar;
colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training_0.png'));
saveas(gcf, fullfile(fpath, 'gmm_training_0.fig'));

figure('position', [100, 100, 600, 600]);
hold on
scatter(X_1(:, 1), X_1(:, 2), 50, 'xk'); 
fcontour(@(x1, x2)(pdf(p_xy_1, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Labeled, class 1');
% colorbar;
colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training_1.png'));
saveas(gcf, fullfile(fpath, 'gmm_training_1.fig'));

%% The marginal distribution p(x), using both labeled and unlabeld data
p_x = @(x1, x2) (p_y_0*pdf(p_xy_0, [x1 x2]) + p_y_1*pdf(p_xy_1, [x1 x2])...
    + p_y_unlabeled*pdf(p_xy_unlabeled, [x1 x2])); 

% Visualize p(x)
figure('position', [100, 100, 600, 600]); hold on;
scatter(X_labeled((y_labeled==0),1), X_labeled((y_labeled==0),2), 50, 'ok');
scatter(X_labeled((y_labeled==1),1), X_labeled((y_labeled==1),2), 50, 'xk');
scatter(X_unlabeled(:, 1), X_unlabeled(:, 2), 50, 's', ...
    'MarkerFaceColor','w','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',1);
fcontour(@(x1, x2)(p_y_0*pdf(p_xy_0, [x1 x2]) + p_y_1*pdf(p_xy_1, [x1 x2])...
    + p_y_unlabeled*pdf(p_xy_unlabeled, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Labeled, class 0', 'Labeled, class 1', 'Unlabeled');
% colorbar;
colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training_marginal.png'));
saveas(gcf, fullfile(fpath, 'gmm_training_marginal.fig'));

%% Testing
y_predict = zeros(N_testing, 1);

for n = 1:N_testing
    this_x_testing = [X_testing(n,1), X_testing(n,2)];
%     w_y1 = h(this_x_testing(1), this_x_testing(2))*pdf(p_xy_1, this_x_testing)/pdf(p_x, this_x_testing);
%     w_y0 = h(this_x_testing(1), this_x_testing(2))*pdf(p_xy_0, this_x_testing)/pdf(p_x, this_x_testing);
    if pdf(p_xy_1, this_x_testing)*p_y_1 >= pdf(p_xy_0, this_x_testing)*p_y_0
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
fcontour(@(x1, x2)(pdf(p_xy_1, [x1, x2])/(p_y_0*pdf(p_xy_0, [x1 x2]) + p_y_1*pdf(p_xy_1, [x1 x2])...
    + p_y_unlabeled*pdf(p_xy_unlabeled, [x1 x2]))*p_y_1 ...
    - pdf(p_xy_0, [x1, x2])/(p_y_0*pdf(p_xy_0, [x1 x2]) + p_y_1*pdf(p_xy_1, [x1 x2])...
    + p_y_unlabeled*pdf(p_xy_unlabeled, [x1 x2]))*p_y_0), ...
    [-5 5 -5 5], '-b', 'LevelList', [0], 'LineWidth', 3)
legend('Correctly labeled 0', 'Correctly labeled 1', 'Errors', 'Decision boundaries');
xlim([-5 5]);
ylim([-5 5]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'testing_decision.png'));
saveas(gcf, fullfile(fpath, 'testing_decision.fig'));

%% Error probability
[X1_test_EP, X2_test_EP] = meshgrid(-5:.1:5);
N_testing_EP = length(X1_test_EP);

tic
y_predict_EP = zeros(N_testing_EP, N_testing_EP); 
for row = 1:N_testing_EP
    for col = 1:N_testing_EP
        this_x_testing = [X1_test_EP(row,col), X2_test_EP(row,col)];
        % Normalization
        this_p_yx_0 = (pdf(p_xy_0, this_x_testing)*p_y_0)/p_x(this_x_testing(1), this_x_testing(2));
        this_p_yx_1 = (pdf(p_xy_1, this_x_testing)*p_y_1)/p_x(this_x_testing(1), this_x_testing(2));
        this_normalizor = 1/(this_p_yx_0 + this_p_yx_1);
        this_p_yx_0 = this_p_yx_0*this_normalizor;
        this_p_yx_1 = this_p_yx_1*this_normalizor;

        if this_p_yx_0 >= this_p_yx_1
            % Decision 0
            y_predict_EP(row, col) = 1 - this_p_yx_0;
        else
            % Decision 1
            y_predict_EP(row, col) = 1 - this_p_yx_1;
        end
    end
end
toc

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
% scatter(X_testing((y_predict==0),1), X_testing((y_predict==0),2), 50, y_predict_EP((y_predict==0)), 'o', 'LineWidth', 3);
% scatter(X_testing((y_predict==1),1), X_testing((y_predict==1),2), 50, y_predict_EP((y_predict==1)), 'x', 'LineWidth', 3);
% scatter(X_testing(find(y_testing - y_predict), 1), X_testing(find(y_testing - y_predict), 2), 100, y_predict_EP(y_testing ~= y_predict), 's', 'filled'); 
contourf(X1_test_EP, X2_test_EP, y_predict_EP);
xlim([-5 5]);
ylim([-5 5]);
colormap(jet)
colorbar;
caxis([0 .5]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'error_probability.png'));
saveas(gcf, fullfile(fpath, 'error_probability.fig'));

%% Monte Carlo Simulations
run simulations_DB.m

run simulations_EP.m

