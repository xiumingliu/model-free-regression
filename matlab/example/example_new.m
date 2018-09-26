clear all
close all

%% Setup 
fpath = 'figures2'; 

N_labeled = 200;    % Number of labeled training data 
N_unlabeled = 1800;  % Number of unlabeld training data

D = 2;      % Dimension of the input X
K = 2;     % Number of components used in GMM

level_list = [0.001 .01 .1:.1:.9 .99 .999];

%% Generate synthetic data
run data_generate.m

%% Histogram of outputs
C = categorical([y_labeled; -1*ones(N_unlabeled, 1)], [0 1 -1],...
    {'Class 0', 'Class 1', 'Unlabeled'});

figure('position', [100, 100, 600, 600]); % Marginal distribution of y
histogram(C)
ylim([0 2000]);
xlabel('$y$', 'Interpreter', 'latex');
ylabel('Histogram of $y$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'histogram.png'));
saveas(gcf, fullfile(fpath, 'histogram.fig'));

%% The conditional distribution of unlabeled data 
% p_y_unlabeled = length(y_unlabeled)/(length(y_labeled) + length(y_unlabeled));    
%     
% [~, model_xy_unlabled, ~] = mixGaussVb(X_unlabeled', K);
% this_Nk = sum(model_xy_unlabled.R);
% this_mu_hat = zeros(D, K);
% for k = 1:K
%     r_nk = model_xy_unlabled.R(:, k);
%     this_mu_hat(:, k) = (1/this_Nk(k))*sum(r_nk.*X_unlabeled);
% end
% this_COV_hat = zeros(D, D, K);
% for k = 1:K
%     r_nk = model_xy_unlabled.R(:, k);
%     this_COV_hat(:, :, k) = (1/this_Nk(k))*((sqrt(r_nk').*(X_unlabeled' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X_unlabeled' - this_mu_hat(:, k)))');
% end
% this_pi_hat = (model_xy_unlabled.alpha/sum(model_xy_unlabled.alpha));
% p_xy_unlabeled = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);
% 
% figure('position', [100, 100, 600, 600]);
% hold on
% scatter(X_unlabeled(:, 1), X_unlabeled(:, 2), 50, 's', 'LineWidth', 1,...
%     'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
% fcontour(@(x1, x2)(pdf(p_xy_unlabeled, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
% legend('Unlabeled', 'Location', 'southeast');
% % colorbar;
% colormap(jet);
% xlim([-5 5]);
% ylim([-5 5]);
% caxis([0 1]);
% xlabel('$x_1$', 'Interpreter', 'latex');
% ylabel('$x_2$', 'Interpreter', 'latex');
% set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'gmm_training_unlabeled.png'));
% saveas(gcf, fullfile(fpath, 'gmm_training_unlabeled.fig'));

%% The conditional distribution p(x | y = 0) and p(x | y = 1), using labeled data
% p(x | y = 0)

y_0 = y_labeled(y_labeled == 0);
X_0 = X_labeled(y_labeled == 0, :);
p_y_0 = length(y_0)/(N_labeled+N_unlabeled);
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
p_y_1 = length(y_1)/(N_labeled+N_unlabeled);
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
scatter(X_0(:, 1), X_0(:, 2), 100, 'or', 'LineWidth', 3); 
fcontour(@(x1, x2)(pdf(p_xy_0, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Labeled, class 0', 'Location', 'southeast');
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
scatter(X_1(:, 1), X_1(:, 2), 100, 'xb', 'LineWidth', 3); 
fcontour(@(x1, x2)(pdf(p_xy_1, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Labeled, class 1', 'Location', 'southeast');
colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training_1.png'));
saveas(gcf, fullfile(fpath, 'gmm_training_1.fig'));

%% The marginal distribution p(x), using labeled and unlabeled data
% p_x = @(x1, x2) (p_y_0*pdf(p_xy_0, [x1 x2]) + p_y_1*pdf(p_xy_1, [x1 x2]) +...
%     p_y_unlabeled*pdf(p_xy_unlabeled, [x1 x2])); 
% 
% 
% % Visualize p(x)
% figure('position', [100, 100, 600, 600]); hold on;
% scatter(X_labeled((y_labeled==0),1), X_labeled((y_labeled==0),2), 100, 'or', 'LineWidth', 3);
% scatter(X_labeled((y_labeled==1),1), X_labeled((y_labeled==1),2), 100, 'xb', 'LineWidth', 3);
% scatter(X_unlabeled(:, 1), X_unlabeled(:, 2), 50, 's', 'LineWidth', 1,...
%     'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
% fcontour(@(x1, x2)(p_y_0*pdf(p_xy_0, [x1 x2]) + p_y_1*pdf(p_xy_1, [x1 x2]) +...
%     p_y_unlabeled*pdf(p_xy_unlabeled, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
% legend('Labeled, class 0', 'Labeled, class 1', 'Unlabeled', 'Location', 'southeast');
% colormap(jet);
% xlim([-5 5]);
% ylim([-5 5]);
% caxis([0 1]);
% xlabel('$x_1$', 'Interpreter', 'latex');
% ylabel('$x_2$', 'Interpreter', 'latex');
% set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'gmm_training_marginal.png'));
% saveas(gcf, fullfile(fpath, 'gmm_training_marginal.fig'));

[~, model_x, ~] = mixGaussVb(X', K);
this_Nk = sum(model_x.R);
this_mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model_x.R(:, k);
    this_mu_hat(:, k) = (1/this_Nk(k))*sum(r_nk.*X);
end
this_COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_x.R(:, k);
    this_COV_hat(:, :, k) = (1/this_Nk(k))*((sqrt(r_nk').*(X' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X' - this_mu_hat(:, k)))');
end
this_pi_hat = (model_x.alpha/sum(model_x.alpha));
p_x = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

figure('position', [100, 100, 600, 600]);
hold on
scatter(X(:, 1), X(:, 2), 50, 's', 'LineWidth', 1,...
    'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
fcontour(@(x1, x2)(pdf(p_x, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Unlabeled', 'Location', 'southeast');
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


%% Posterior probability for unlabeled data
% [X1_test_EP, X2_test_EP] = meshgrid(-5:.1:5);
% N_testing_EP = length(X1_test_EP);
alpha = 0.2;

tic
y_predict = zeros(N_unlabeled, 1);
y_predict_0 = zeros(N_unlabeled, 1); 
y_predict_1 = zeros(N_unlabeled, 1); 
for n = 1:N_unlabeled
    
        this_x_unlabeled = X_unlabeled(n, :);
%         this_p_yx_0 = (pdf(p_xy_0, this_x_unlabeled)*.5)/p_x(this_x_unlabeled(1), this_x_unlabeled(2));
%         this_p_yx_1 = (pdf(p_xy_1, this_x_unlabeled)*.5)/p_x(this_x_unlabeled(1), this_x_unlabeled(2));
        this_p_yx_0 = (pdf(p_xy_0, this_x_unlabeled)*.5)/pdf(p_x, this_x_unlabeled);
        this_p_yx_1 = (pdf(p_xy_1, this_x_unlabeled)*.5)/pdf(p_x, this_x_unlabeled);

        y_predict_0(n) = this_p_yx_0;
        y_predict_1(n) = this_p_yx_1;

        if y_predict_0(n) >= 1 - alpha &&  y_predict_1(n) < 1 - alpha
            y_predict(n) = 0;
        elseif y_predict_0(n) < 1 - alpha &&  y_predict_1(n) >= 1 - alpha
            y_predict(n) = 1;
        else
            y_predict(n) = round(rand);
        end

end
toc

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
scatter(X_unlabeled(y_predict == 0, 1), X_unlabeled(y_predict == 0, 2), 100, 'or', 'LineWidth', 3);
scatter(X_unlabeled(y_predict == 1, 1), X_unlabeled(y_predict == 1, 2), 100, 'xb', 'LineWidth', 3);
xlim([-5 5]);
ylim([-5 5]);
colormap(jet)
colorbar;
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'labeling.png'));
saveas(gcf, fullfile(fpath, 'labeling.fig'));

%% New probabilities
C = categorical([y_labeled; y_predict], [0 1 -1],...
    {'Class 0', 'Class 1', 'Unlabeled'});

figure('position', [100, 100, 600, 600]); % Marginal distribution of y
histogram(C)
ylim([0 2000]);
xlabel('$y$', 'Interpreter', 'latex');
ylabel('Histogram of $y$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'histogram_new.png'));
saveas(gcf, fullfile(fpath, 'histogram_new.fig'));

% p'(x | y = 0)
y_0 = [y_labeled(y_labeled == 0); y_predict(y_predict == 0)];
X_0 = [X_labeled(y_labeled == 0, :); X_unlabeled(y_predict == 0, :)];
p_y_0_new = length(y_0)/(N_labeled+N_unlabeled);
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
p_xy_0_new = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

% p'(x | y = 1)
y_1 = [y_labeled(y_labeled == 1); y_predict(y_predict == 1)];
X_1 = [X_labeled(y_labeled == 1, :); X_unlabeled(y_predict == 1, :)];
p_y_1_new = length(y_1)/(N_labeled+N_unlabeled);
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
p_xy_1_new = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

figure('position', [100, 100, 600, 600]);
hold on
scatter(X_0(:, 1), X_0(:, 2), 100, 'or', 'LineWidth', 3); 
fcontour(@(x1, x2)(pdf(p_xy_0_new, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Labeled, class 0', 'Location', 'southeast');
colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training_0_new.png'));
saveas(gcf, fullfile(fpath, 'gmm_training_0.fig'));

figure('position', [100, 100, 600, 600]);
hold on
scatter(X_1(:, 1), X_1(:, 2), 100, 'xb', 'LineWidth', 3); 
fcontour(@(x1, x2)(pdf(p_xy_1_new, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Labeled, class 1', 'Location', 'southeast');
colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training_1_new.png'));
saveas(gcf, fullfile(fpath, 'gmm_training_1_new.fig'));

% % p'(x)
p_x_new = @(x1, x2) (p_y_0_new*pdf(p_xy_0_new, [x1 x2]) + p_y_1_new*pdf(p_xy_1_new, [x1 x2])); 

%% Error probability
[X1_test_EP, X2_test_EP] = meshgrid(-5:.2:5);
N_testing_EP = length(X1_test_EP);

tic
y_predict_EP = zeros(N_testing_EP, N_testing_EP); 

y_predict_P = zeros(N_testing_EP, N_testing_EP); 
for row = 1:N_testing_EP
    for col = 1:N_testing_EP
        this_x_testing = [X1_test_EP(row,col), X2_test_EP(row,col)];
        this_p_yx_0 = (pdf(p_xy_0_new, this_x_testing)*p_y_0_new)/p_x_new(this_x_testing(1), this_x_testing(2));
        this_p_yx_1 = (pdf(p_xy_1_new, this_x_testing)*p_y_1_new)/p_x_new(this_x_testing(1), this_x_testing(2));
        
%         this_p_yx_0 = (pdf(p_xy_0_new, this_x_testing)*p_y_0_new)/pdf(p_x, this_x_testing);
%         this_p_yx_1 = (pdf(p_xy_1_new, this_x_testing)*p_y_1_new)/pdf(p_x, this_x_testing);


        if this_p_yx_0 >= this_p_yx_1
            % Decision 0
            y_predict_EP(row, col) = 1 - this_p_yx_0;
            y_predict_P(row, col) = this_p_yx_0;
%             y_predict_EP(row, col) = this_p_yx_1;
        else
            % Decision 1
            y_predict_EP(row, col) = 1 - this_p_yx_1;
            y_predict_P(row, col) = this_p_yx_1;
%             y_predict_EP(row, col) = this_p_yx_0;
        end
    end
end
toc

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
contourf(X1_test_EP, X2_test_EP, y_predict_EP);
xlim([-3.5 3.5]);
ylim([-3.5 3.5]);
colormap(jet)
colorbar;
caxis([0 .5]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'error_probability.png'));
saveas(gcf, fullfile(fpath, 'error_probability.fig'));

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
contourf(X1_test_EP, X2_test_EP, y_predict_P);
xlim([-3.5 3.5]);
ylim([-3.5 3.5]);
colormap(jet)
colorbar;
caxis([.5 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'success_probability.png'));
saveas(gcf, fullfile(fpath, 'success_probability.fig'));
