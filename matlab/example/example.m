clear all
close all

%% Setup 
fpath = 'figures1'; 

N_labeled = 100;    % Number of labeled training data 
N_unlabeled = 900;  % Number of unlabeld training data

D = 2;      % Dimension of the input X
K = 50;     % Number of components used in GMM

level_list = [0.001 .01 .1:.1:.9 .99 .999];

%% Generate synthetic data
run data_generate.m

%% Histogram of outputs
C = categorical([y_labeled; -1*ones(N_unlabeled, 1)], [0 1 -1],...
    {'Class 0', 'Class 1', 'Unlabeled'});

figure('position', [100, 100, 600, 600]); % Marginal distribution of y
histogram(C)
ylim([0 1000]);
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
% K = 1;

y_0 = y_labeled(y_labeled == 0);
X_0 = X_labeled(y_labeled == 0, :);
p_y_0 = length(y_0)/(length(y_labeled));
% p_y_0 = length(y_0)/(length(y_labeled) + length(y_unlabeled));
% p_y_0 = (length(y_labeled)/2)/(length(y_labeled) + length(y_unlabeled));
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
p_y_1 = length(y_1)/(length(y_labeled));
% p_y_1 = (length(y_labeled)/2)/(length(y_labeled) + length(y_unlabeled));
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
scatter(X_1(:, 1), X_1(:, 2), 100, 'xb', 'LineWidth', 3); 
fcontour(@(x1, x2)(pdf(p_xy_1, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Labeled, class 1', 'Location', 'southeast');
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

%% The marginal distribution p(x), using labeled data
p_x = @(x1, x2) (p_y_0*pdf(p_xy_0, [x1 x2]) + p_y_1*pdf(p_xy_1, [x1 x2])); 

% Visualize p(x)
figure('position', [100, 100, 600, 600]); hold on;
scatter(X_labeled((y_labeled==0),1), X_labeled((y_labeled==0),2), 100, 'or', 'LineWidth', 3);
scatter(X_labeled((y_labeled==1),1), X_labeled((y_labeled==1),2), 100, 'xb', 'LineWidth', 3);
scatter(X_unlabeled(:, 1), X_unlabeled(:, 2), 50, 's', 'LineWidth', 1,...
    'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
fcontour(@(x1, x2)(p_y_0*pdf(p_xy_0, [x1 x2]) + p_y_1*pdf(p_xy_1, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
legend('Labeled, class 0', 'Labeled, class 1', 'Unlabeled', 'Location', 'southeast');
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


%% Labeling

% Error Tolenrance 
alpha = 1e-12; 

% p_x_labeled = @(x1, x2) (.5*pdf(p_xy_0, [x1 x2]) + .5*pdf(p_xy_1, [x1 x2])); 
X_unlabeled_labeled = [];
y_unlabeled_labeled = [];
for n = 1:N_unlabeled
    this_x_unlabeled = [X_unlabeled(n, 1), X_unlabeled(n, 2)];
    
    this_p_yx_0 = (pdf(p_xy_0, this_x_unlabeled)*p_y_0)/p_x(this_x_unlabeled(1), this_x_unlabeled(2));
    this_p_yx_1 = (pdf(p_xy_1, this_x_unlabeled)*p_y_1)/p_x(this_x_unlabeled(1), this_x_unlabeled(2));
    
    if this_p_yx_1 > 1 - alpha && this_p_yx_0 < 1 - alpha
        X_unlabeled_labeled = [X_unlabeled_labeled; this_x_unlabeled];
        y_unlabeled_labeled = [y_unlabeled_labeled; 1];
    elseif this_p_yx_0 > 1 - alpha && this_p_yx_1 < 1 - alpha
        X_unlabeled_labeled = [X_unlabeled_labeled; this_x_unlabeled];
        y_unlabeled_labeled = [y_unlabeled_labeled; 0];
    end
end

[X_unlabeled_unlabeled, index] = setdiff(X_unlabeled, X_unlabeled_labeled, 'row');
y_unlabeled_unlabeled = y_unlabeled(index); 

% Visualize labeled unlabeled data
figure('position', [100, 100, 600, 600]); hold on;
scatter(X_unlabeled_labeled((y_unlabeled_labeled==0),1), X_unlabeled_labeled((y_unlabeled_labeled==0),2), 100, 'or', 'LineWidth', 3);
scatter(X_unlabeled_labeled((y_unlabeled_labeled==1),1), X_unlabeled_labeled((y_unlabeled_labeled==1),2), 100, 'xb', 'LineWidth', 3);
scatter(X_unlabeled_unlabeled(:, 1), X_unlabeled_unlabeled(:, 2), 50, 's', 'LineWidth', 1,...
    'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
legend('New labeled, class 0', 'New labeled, class 1', 'Unlabeled', 'Location', 'southeast');
% colorbar;
colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'gmm_training_new.png'));
saveas(gcf, fullfile(fpath, 'gmm_training_new.fig'));


X_labeled_new = [X_labeled; X_unlabeled_labeled];
y_labeled_new = [y_labeled; y_unlabeled_labeled]; 

X_unlabeled_new = X_unlabeled_unlabeled;
y_unlabeled_new = y_unlabeled_unlabeled;

% %% New p(x), p(x | y = 1, 0), p(x | u)
% % p'(x | y = 0)
% y_0 = y_labeled_new(y_labeled_new == 0);
% X_0 = X_labeled_new(y_labeled_new == 0, :);
% p_y_0 = length(y_0)/(length(y_labeled_new) + length(y_unlabeled));
% [~, model_xy_0, ~] = mixGaussVb(X_0', K);
% this_Nk = sum(model_xy_0.R);
% this_mu_hat = zeros(D, K);
% for k = 1:K
%     r_nk = model_xy_0.R(:, k);
%     this_mu_hat(:, k) = (1/this_Nk(k))*sum(r_nk.*X_0);
% end
% this_COV_hat = zeros(D, D, K);
% for k = 1:K
%     r_nk = model_xy_0.R(:, k);
%     this_COV_hat(:, :, k) = (1/this_Nk(k))*((sqrt(r_nk').*(X_0' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X_0' - this_mu_hat(:, k)))');
% end
% this_pi_hat = (model_xy_0.alpha/sum(model_xy_0.alpha));
% p_xy_0 = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);
% 
% % p'(x | y = 1)
% y_1 = y_labeled_new(y_labeled_new == 1);
% X_1 = X_labeled_new(y_labeled_new == 1, :);
% p_y_1 = length(y_1)/length(y);
% [~, model_xy_1, ~] = mixGaussVb(X_1', K);
% this_N_k = sum(model_xy_1.R);
% this_mu_hat = zeros(D, K);
% for k = 1:K
%     r_nk = model_xy_1.R(:, k);
%     this_mu_hat(:, k) = (1/this_N_k(k))*sum(r_nk.*X_1);
% end
% this_COV_hat = zeros(D, D, K);
% for k = 1:K
%     r_nk = model_xy_1.R(:, k);
%     this_COV_hat(:, :, k) = (1/this_N_k(k))*((sqrt(r_nk').*(X_1' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X_1' - this_mu_hat(:, k)))');
% end
% this_pi_hat = (model_xy_1.alpha/sum(model_xy_1.alpha));
% p_xy_1 = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);
% 
% figure('position', [100, 100, 600, 600]);
% hold on
% scatter(X_0(:, 1), X_0(:, 2), 100, 'or', 'LineWidth', 3); 
% fcontour(@(x1, x2)(pdf(p_xy_0, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
% legend('Labeled, class 0', 'Location', 'southeast');
% % colorbar;
% colormap(jet);
% xlim([-5 5]);
% ylim([-5 5]);
% caxis([0 1]);
% xlabel('$x_1$', 'Interpreter', 'latex');
% ylabel('$x_2$', 'Interpreter', 'latex');
% set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'gmm_training_0_new.png'));
% saveas(gcf, fullfile(fpath, 'gmm_training_0_new.fig'));
% 
% figure('position', [100, 100, 600, 600]);
% hold on
% scatter(X_1(:, 1), X_1(:, 2), 100, 'xb', 'LineWidth', 3); 
% fcontour(@(x1, x2)(pdf(p_xy_1, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
% legend('Labeled, class 1', 'Location', 'southeast');
% % colorbar;
% colormap(jet);
% xlim([-5 5]);
% ylim([-5 5]);
% caxis([0 1]);
% xlabel('$x_1$', 'Interpreter', 'latex');
% ylabel('$x_2$', 'Interpreter', 'latex');
% set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'gmm_training_1_new.png'));
% saveas(gcf, fullfile(fpath, 'gmm_training_1_new.fig'));
% 
% % p'(x | u)
% p_y_unlabeled_new = length(y_unlabeled_new)/(length(y_labeled_new) + length(y_unlabeled_new));    
%     
% [~, model_xy_unlabled, ~] = mixGaussVb(X_unlabeled_new', K);
% this_Nk = sum(model_xy_unlabled.R);
% this_mu_hat = zeros(D, K);
% for k = 1:K
%     r_nk = model_xy_unlabled.R(:, k);
%     this_mu_hat(:, k) = (1/this_Nk(k))*sum(r_nk.*X_unlabeled_new);
% end
% this_COV_hat = zeros(D, D, K);
% for k = 1:K
%     r_nk = model_xy_unlabled.R(:, k);
%     this_COV_hat(:, :, k) = (1/this_Nk(k))*((sqrt(r_nk').*(X_unlabeled_new' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X_unlabeled_new' - this_mu_hat(:, k)))');
% end
% this_pi_hat = (model_xy_unlabled.alpha/sum(model_xy_unlabled.alpha));
% p_xy_unlabeled_new = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);
% 
% figure('position', [100, 100, 600, 600]);
% hold on
% scatter(X_unlabeled_new(:, 1), X_unlabeled_new(:, 2), 50, 's', 'LineWidth', 1,...
%     'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
% fcontour(@(x1, x2)(pdf(p_xy_unlabeled_new, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
% legend('Unlabeled', 'Location', 'southeast');
% % colorbar;
% colormap(jet);
% xlim([-5 5]);
% ylim([-5 5]);
% caxis([0 1]);
% xlabel('$x_1$', 'Interpreter', 'latex');
% ylabel('$x_2$', 'Interpreter', 'latex');
% set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'gmm_training_unlabeled_new.png'));
% saveas(gcf, fullfile(fpath, 'gmm_training_unlabeled_new.fig'));
% 
% % p'(x)
% p_x = @(x1, x2) (p_y_0*pdf(p_xy_0, [x1 x2]) + p_y_1*pdf(p_xy_1, [x1 x2])...
%     + p_y_unlabeled_new*pdf(p_xy_unlabeled_new, [x1 x2])); 
% 
% % Visualize p(x)
% figure('position', [100, 100, 600, 600]); hold on;
% scatter(X_labeled_new((y_labeled_new==0),1), X_labeled_new((y_labeled_new==0),2), 100, 'or', 'LineWidth', 3);
% scatter(X_labeled_new((y_labeled_new==1),1), X_labeled_new((y_labeled_new==1),2), 100, 'xb', 'LineWidth', 3);
% scatter(X_unlabeled_new(:, 1), X_unlabeled_new(:, 2), 50, 's', 'LineWidth', 1,...
%     'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
% fcontour(@(x1, x2)(p_y_0*pdf(p_xy_0, [x1 x2]) + p_y_1*pdf(p_xy_1, [x1 x2])...
%     + p_y_unlabeled_new*pdf(p_xy_unlabeled_new, [x1 x2])), [-5 5 -5 5], 'LevelList', level_list, 'LineWidth', 1)
% legend('Labeled, class 0', 'Labeled, class 1', 'Unlabeled', 'Location', 'southeast');
% % colorbar;
% colormap(jet);
% xlim([-5 5]);
% ylim([-5 5]);
% caxis([0 1]);
% xlabel('$x_1$', 'Interpreter', 'latex');
% ylabel('$x_2$', 'Interpreter', 'latex');
% set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'gmm_training_marginal_new.png'));
% saveas(gcf, fullfile(fpath, 'gmm_training_marginal_new.fig'));
% 
% %% Error probability without normalization
% [X1_test_EP, X2_test_EP] = meshgrid(-5:.1:5);
% N_testing_EP = length(X1_test_EP);
% 
% tic
% y_predict_EP = zeros(N_testing_EP, N_testing_EP); 
% for row = 1:N_testing_EP
%     for col = 1:N_testing_EP
%         this_x_testing = [X1_test_EP(row,col), X2_test_EP(row,col)];
%         % Normalization
% %         this_p_yx_0 = (pdf(p_xy_0, this_x_testing)*p_y_0)/p_x(this_x_testing(1), this_x_testing(2));
% %         this_p_yx_1 = (pdf(p_xy_1, this_x_testing)*p_y_1)/p_x(this_x_testing(1), this_x_testing(2));
% 
%         this_p_yx_0 = (pdf(p_xy_0, this_x_testing)*.5)/p_x(this_x_testing(1), this_x_testing(2));
%         this_p_yx_1 = (pdf(p_xy_1, this_x_testing)*.5)/p_x(this_x_testing(1), this_x_testing(2));
% % 
% %         this_normalizor = 1/(this_p_yx_0 + this_p_yx_1);
% %         this_p_yx_0 = this_p_yx_0*this_normalizor;
% %         this_p_yx_1 = this_p_yx_1*this_normalizor;
% 
%         if this_p_yx_0 >= this_p_yx_1
%             % Decision 0
% %             y_predict_EP(row, col) = 1 - this_p_yx_0;
%             y_predict_EP(row, col) = this_p_yx_1;
%         else
%             % Decision 1
% %             y_predict_EP(row, col) = 1 - this_p_yx_1;
%             y_predict_EP(row, col) = this_p_yx_0;
%         end
%     end
% end
% toc
% 
% figure('position', [100, 100, 600, 600]); % Scatter plot 
% hold on;
% % scatter(X_testing((y_predict==0),1), X_testing((y_predict==0),2), 50, y_predict_EP((y_predict==0)), 'o', 'LineWidth', 3);
% % scatter(X_testing((y_predict==1),1), X_testing((y_predict==1),2), 50, y_predict_EP((y_predict==1)), 'x', 'LineWidth', 3);
% % scatter(X_testing(find(y_testing - y_predict), 1), X_testing(find(y_testing - y_predict), 2), 100, y_predict_EP(y_testing ~= y_predict), 's', 'filled'); 
% contourf(X1_test_EP, X2_test_EP, y_predict_EP);
% xlim([-5 5]);
% ylim([-5 5]);
% colormap(jet)
% colorbar;
% caxis([0 .5]);
% xlabel('$x_1$', 'Interpreter', 'latex');
% ylabel('$x_2$', 'Interpreter', 'latex');
% set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'error_probability_2.png'));
% saveas(gcf, fullfile(fpath, 'error_probability_2.fig'));
% 
% %% Correct probability without normalization
% [X1_test_P, X2_test_P] = meshgrid(-5:.1:5);
% N_testing_P = length(X1_test_P);
% 
% tic
% y_predict_P = zeros(N_testing_P, N_testing_P); 
% for row = 1:N_testing_P
%     for col = 1:N_testing_P
%         this_x_testing = [X1_test_P(row,col), X2_test_P(row,col)];
%         % Normalization
%         this_p_yx_0 = (pdf(p_xy_0, this_x_testing)*p_y_0)/p_x(this_x_testing(1), this_x_testing(2));
%         this_p_yx_1 = (pdf(p_xy_1, this_x_testing)*p_y_1)/p_x(this_x_testing(1), this_x_testing(2));
% 
% %         this_p_yx_0 = (pdf(p_xy_0, this_x_testing)*.5)/p_x(this_x_testing(1), this_x_testing(2));
% %         this_p_yx_1 = (pdf(p_xy_1, this_x_testing)*.5)/p_x(this_x_testing(1), this_x_testing(2));
% 
%         if this_p_yx_0 >= this_p_yx_1
%             % Decision 0
%             y_predict_P(row, col) = this_p_yx_0;
%         else
%             % Decision 1
%             y_predict_P(row, col) = this_p_yx_1;
%         end
%     end
% end
% toc
% 
% figure('position', [100, 100, 600, 600]); % Scatter plot 
% hold on;
% % scatter(X_testing((y_predict==0),1), X_testing((y_predict==0),2), 50, y_predict_P((y_predict==0)), 'o', 'LineWidth', 3);
% % scatter(X_testing((y_predict==1),1), X_testing((y_predict==1),2), 50, y_predict_P((y_predict==1)), 'x', 'LineWidth', 3);
% % scatter(X_testing(find(y_testing - y_predict), 1), X_testing(find(y_testing - y_predict), 2), 100, y_predict_P(y_testing ~= y_predict), 's', 'filled'); 
% contourf(X1_test_P, X2_test_P, y_predict_P);
% xlim([-5 5]);
% ylim([-5 5]);
% colormap(jet)
% colorbar;
% caxis([0 1]);
% xlabel('$x_1$', 'Interpreter', 'latex');
% ylabel('$x_2$', 'Interpreter', 'latex');
% set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'correct_probability.png'));
% saveas(gcf, fullfile(fpath, 'correct_probability.fig'));
% 
% 
% 
