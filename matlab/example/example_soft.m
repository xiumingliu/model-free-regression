clear all
close all

%% Setup 
fpath = 'figures7'; 

N_labeled = 200;    % Number of labeled training data 
N_unlabeled = 3000;  % Number of unlabeld training data

D = 2;      % Dimension of the input X
K = 2;     % Number of components used in GMM

level_list = [0.001 .01 .1:.1:.9 .99 .999];

%% Generate synthetic data
run data_generate.m

%% The conditional distribution of unlabeled data 
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
scatter(X_unlabeled(:, 1), X_unlabeled(:, 2), 50, 's', 'LineWidth', 1,...
    'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
fcontour(@(x1, x2)(pdf(p_xy_unlabeled, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Unlabeled', 'Location', 'southeast');
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
K = 1;
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
K = 1;
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
K = 2;
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
legend('Marginal', 'Location', 'southeast');
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


%% LRT for unlabeled data
[X1_test_LRT, X2_test_LRT] = meshgrid(-5:.1:5);
N_testing_LRT = length(X1_test_LRT);
alpha = 0.01;

y_LRT = zeros(N_testing_LRT, N_testing_LRT);
tic
for row = 1:N_testing_LRT
    for col = 1:N_testing_LRT
        this_x_testing = [X1_test_LRT(row,col), X2_test_LRT(row,col)];
        y_LRT(row, col) = log(pdf(p_xy_unlabeled, this_x_testing)/...
            (.5*pdf(p_xy_0, this_x_testing) + .5*pdf(p_xy_1, this_x_testing)));
    end
end
toc

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
imagesc(-5:.1:5, -5:.1:5, y_LRT);
% contour(-5:.1:5, -5:.1:5, y_LRT, 'LevelList', [0])
xlim([-5 5]);
ylim([-5 5]);
colormap(gray)
colorbar;
caxis([-1 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'lrt.png'));
saveas(gcf, fullfile(fpath, 'lrt.fig'));

tic
log_likelihood_ratio = zeros(N_unlabeled, 1); 
X_labeled_augment = X_labeled;
y_labeled_augment = y_labeled;
X_unlabeled_remain = X_unlabeled;
y_unlabeled_remain = y_unlabeled;
for n = 1:N_unlabeled
    
        this_x_unlabeled = X_unlabeled(n, :);
        
        log_likelihood_ratio(n) = log(pdf(p_xy_unlabeled, this_x_unlabeled)/...
            (.5*pdf(p_xy_0, this_x_unlabeled) + .5*pdf(p_xy_1, this_x_unlabeled)));
        
        if log_likelihood_ratio(n) < 0 
            X_labeled_augment = [X_labeled_augment; this_x_unlabeled]; 
            index = X_unlabeled_remain ~= this_x_unlabeled;
            X_unlabeled_remain = X_unlabeled_remain(index(:, 1), :);
            y_unlabeled_remain = y_unlabeled_remain(index(:, 1));
            if pdf(p_xy_0, this_x_unlabeled) > pdf(p_xy_1, this_x_unlabeled)
%                 y_predict(n) = 0;
                y_labeled_augment = [y_labeled_augment; 0];
            else
%                 y_predict(n) = 1;
                y_labeled_augment = [y_labeled_augment; 1];
            end
        end
end
toc

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
scatter(X_labeled_augment(y_labeled_augment == 0, 1), X_labeled_augment(y_labeled_augment == 0, 2), 100, 'or', 'LineWidth', 3);
scatter(X_labeled_augment(y_labeled_augment == 1, 1), X_labeled_augment(y_labeled_augment == 1, 2), 100, 'xb', 'LineWidth', 3);
scatter(X_unlabeled_remain(y_unlabeled_remain == 0, 1), X_unlabeled_remain(y_unlabeled_remain == 0, 2), 100, 'o', 'LineWidth', 3,...
    'MarkerFaceColor','w','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
scatter(X_unlabeled_remain(y_unlabeled_remain == 1, 1), X_unlabeled_remain(y_unlabeled_remain == 1, 2), 100, 'x', 'LineWidth', 3,...
    'MarkerFaceColor','w','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
xlim([-5 5]);
ylim([-5 5]);
% colormap(jet)
% colorbar;
% caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'labeling.png'));
saveas(gcf, fullfile(fpath, 'labeling.fig'));

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
scatter(X_unlabeled(:, 1), X_unlabeled(:, 2), 100, log_likelihood_ratio, 'LineWidth', 3);
xlim([-5 5]);
ylim([-5 5]);
colormap(jet)
colorbar;
caxis([-1 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'likelihood.png'));
saveas(gcf, fullfile(fpath, 'likelihood.fig'));

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
histogram(log_likelihood_ratio,'Normalization', 'pdf');
plot(zeros(length(0:.01:0.35), 1), 0:.01:0.35, 'LineWidth', 3);
% caxis([0 1]);
xlabel('Log-likelihood ratio', 'Interpreter', 'latex');
ylabel('Hitogram', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'likelihood_hist.png'));
saveas(gcf, fullfile(fpath, 'likelihood_hist.fig'));

%% New probabilities
% p'(x | y = 0)
K = 10; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_0 = y_labeled_augment(y_labeled_augment == 0);
X_0 = X_labeled_augment(y_labeled_augment == 0, :);
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
y_1 = y_labeled_augment(y_labeled_augment == 1); 
X_1 = X_labeled_augment(y_labeled_augment == 1, :); 
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

% p'(x | y = u)
p_y_unlabeled_new = length(y_unlabeled_remain)/(length(y_labeled) + length(y_unlabeled));    
K = 30;    
[~, model_xy_unlabled, ~] = mixGaussVb(X_unlabeled_remain', K);
this_Nk = sum(model_xy_unlabled.R);
this_mu_hat = zeros(D, K);
for k = 1:K
    r_nk = model_xy_unlabled.R(:, k);
    this_mu_hat(:, k) = (1/this_Nk(k))*sum(r_nk.*X_unlabeled_remain);
end
this_COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_xy_unlabled.R(:, k);
    this_COV_hat(:, :, k) = (1/this_Nk(k))*((sqrt(r_nk').*(X_unlabeled_remain' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X_unlabeled_remain' - this_mu_hat(:, k)))');
end
this_pi_hat = (model_xy_unlabled.alpha/sum(model_xy_unlabled.alpha));
p_xy_unlabeled_new = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

figure('position', [100, 100, 600, 600]);
hold on
scatter(X_unlabeled_remain(:, 1), X_unlabeled_remain(:, 2), 50, 's', 'LineWidth', 1,...
    'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
fcontour(@(x1, x2)(pdf(p_xy_unlabeled_new, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Unlabeled', 'Location', 'southeast');
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

% % p'(x)
p_x_new = @(x1, x2) (p_y_0_new*pdf(p_xy_0_new, [x1 x2]) + p_y_1_new*pdf(p_xy_1_new, [x1 x2]) + p_y_unlabeled_new*pdf(p_xy_unlabeled_new, [x1 x2])); 

%% The Example
% MCAR
% P(x | y = 1) 
p_xy_1_mcar.m = p_x.mu(1, :);
p_xy_1_mcar.cov = p_x.Sigma(:, :, 1);

p_xy_0_mcar.m = p_x.mu(2, :);
p_xy_0_mcar.cov = p_x.Sigma(:, :, 2);

% Figure
figure('position', [100, 100, 1000, 600]);
hold on
scatter(X_unlabeled((y_unlabeled==0), 1), X_unlabeled((y_unlabeled==0), 2), 100, 'o', 'LineWidth', 3,...
    'MarkerFaceColor','w','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
scatter(X_unlabeled((y_unlabeled==1), 1), X_unlabeled((y_unlabeled==1), 2), 100, 'x', 'LineWidth', 3,...
    'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
scatter(X_labeled((y_labeled==0),1), X_labeled((y_labeled==0),2), 100, 'or', 'LineWidth', 3);
scatter(X_labeled((y_labeled==1),1), X_labeled((y_labeled==1),2), 100, 'xb', 'LineWidth', 3);
fcontour(@(x1, x2)(p_y_1*pdf(p_xy_1, [x1 x2]) - p_y_0*pdf(p_xy_0, [x1 x2])),...
    [-5 5 -5 5], 'LevelList', [0], 'LineWidth', 3, 'LineStyle', '--', 'LineColor', 'k')
fcontour(@(x1, x2)(p_y_1*mvnpdf([x1 x2], p_xy_1_mcar.m, p_xy_1_mcar.cov) - ...
    p_y_0*mvnpdf([x1 x2], p_xy_0_mcar.m, p_xy_0_mcar.cov)), [-5 5 -5 5], ...
    'LevelList', [0], 'LineWidth', 3, 'LineStyle', ':', 'LineColor', 'k')

plot(zeros(length(-5:.1:5), 1), -5:.1:5, '-k', 'LineWidth', 3);

lgd = legend({'Labeled, class 0', 'Labeled, class 1', 'Unlabeled, class 0', ...
    'Unlabeled, class 1', 'Supervised (VB-GMM)', 'Semi-supervised (Self Training)', 'True boundary'}, ...
    'Location', 'westoutside', 'Interpreter', 'latex');
% colorbar;
% colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 20, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'data.png'));
saveas(gcf, fullfile(fpath, 'data.fig'));

%% testing

tic
y_uncertainty = zeros(N_unlabeled, 1);
y_predict= zeros(N_unlabeled, 1); 
p_y0_x = zeros(N_unlabeled, 1); 
for n = 1:N_unlabeled
    
        this_x_unlabeled = X_unlabeled(n, :);
        this_p_yx_0 = (pdf(p_xy_0_new, this_x_unlabeled)*p_y_0_new)/p_x_new(this_x_unlabeled(1), this_x_unlabeled(2));
        this_p_yx_1 = (pdf(p_xy_1_new, this_x_unlabeled)*p_y_1_new)/p_x_new(this_x_unlabeled(1), this_x_unlabeled(2));
        this_p_yx_unlabeled = (pdf(p_xy_unlabeled_new, this_x_unlabeled)*p_y_unlabeled_new)/p_x_new(this_x_unlabeled(1), this_x_unlabeled(2));
        p_y0_x(n) = this_p_yx_0 + this_p_yx_unlabeled*.5;

end
toc

% [X1_test_EP_1, X2_test_EP_1] = meshgrid(-5:.2:.2);
% N_testing_EP_1 = length(X1_test_EP_1);
% 
% tic
% y_predict_EP_1 = zeros(N_testing_EP_1, N_testing_EP_1); 
% for row = 1:N_testing_EP_1
%     for col = 1:N_testing_EP_1
%         this_x_testing = [X1_test_EP_1(row,col), X2_test_EP_1(row,col)];
%         this_p_yx_0 = (pdf(p_xy_0_new, this_x_testing)*p_y_0_new)/p_x_new(this_x_testing(1), this_x_testing(2));
%         this_p_yx_1 = (pdf(p_xy_1_new, this_x_testing)*p_y_1_new)/p_x_new(this_x_testing(1), this_x_testing(2));
%         this_p_yx_unlabeled = (pdf(p_xy_unlabeled_new, this_x_unlabeled)*p_y_unlabeled_new)/p_x_new(this_x_unlabeled(1), this_x_unlabeled(2));
%         if this_p_yx_0 >= this_p_yx_1
%             % Decision 0
%             y_predict_EP_1(row, col) = 1 - this_p_yx_0 - .5*this_p_yx_unlabeled;
%         else
%             % Decision 1
%             y_predict_EP_1(row, col) = 1 - this_p_yx_1 - .5*this_p_yx_unlabeled;
%         end
%     end
% end
% toc
% 
% [X1_test_EP_2, X2_test_EP_2] = meshgrid(-.2:.2:5);
% N_testing_EP_2 = length(X1_test_EP_2);
% 
% tic
% y_predict_EP_2 = zeros(N_testing_EP_2, N_testing_EP_2); 
% for row = 1:N_testing_EP_2
%     for col = 1:N_testing_EP_2
%         this_x_testing = [X1_test_EP_2(row,col), X2_test_EP_2(row,col)];
%         this_p_yx_0 = (pdf(p_xy_0_new, this_x_testing)*p_y_0_new)/p_x_new(this_x_testing(1), this_x_testing(2));
%         this_p_yx_1 = (pdf(p_xy_1_new, this_x_testing)*p_y_1_new)/p_x_new(this_x_testing(1), this_x_testing(2));
%         this_p_yx_unlabeled = (pdf(p_xy_unlabeled_new, this_x_unlabeled)*p_y_unlabeled_new)/p_x_new(this_x_unlabeled(1), this_x_unlabeled(2));
%         if this_p_yx_0 >= this_p_yx_1
%             % Decision 0
%             y_predict_EP_2(row, col) = 1 - this_p_yx_0 - .5*this_p_yx_unlabeled;
%         else
%             % Decision 1
%             y_predict_EP_2(row, col) = 1 - this_p_yx_1 - .5*this_p_yx_unlabeled;
%         end
%     end
% end
% toc

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
scatter(X_unlabeled(:, 1), X_unlabeled(:, 2), 50,...
    p_y0_x, 'filled', 'o', 'LineWidth', 3, 'MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5);
% contour(X1_test_EP_1, X2_test_EP_1, y_predict_EP_1, 'LevelList', [0.2], 'LineWidth', 3, 'LineColor', 'k');
% contour(X1_test_EP_2, X2_test_EP_2, y_predict_EP_2, 'LevelList', [0.2], 'LineWidth', 3, 'LineColor', 'k');
xlim([-5 5]);
ylim([-5 5]);
colormap(jet)
colorbar;
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 20, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'p_y0_x.png'));
saveas(gcf, fullfile(fpath, 'p_y0_x.fig'));
