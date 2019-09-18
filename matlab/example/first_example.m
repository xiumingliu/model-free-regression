clear all
close all

%% Setup 
fpath = 'figures8'; 

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
            p_y_1_x = pdf(p_xy_1, this_x_unlabeled)/(pdf(p_xy_1, this_x_unlabeled) + pdf(p_xy_0, this_x_unlabeled));
%             if pdf(p_xy_0, this_x_unlabeled) > pdf(p_xy_1, this_x_unlabeled)
% %                 y_predict(n) = 0;
%                 y_labeled_augment = [y_labeled_augment; 0];
%             else
% %                 y_predict(n) = 1;
%                 y_labeled_augment = [y_labeled_augment; 1];
%             end
            y_labeled_augment = [y_labeled_augment; binornd(1, p_y_1_x)];
        end
end
toc

%% MCAR
tic
X_labeled_augment_mcar = X_labeled;
y_labeled_augment_mcar = y_labeled;
X_unlabeled_remain_mcar = X_unlabeled;
y_unlabeled_remain_mcar = y_unlabeled;
for n = 1:N_unlabeled
    
    this_x_unlabeled = X_unlabeled(n, :);
        
       
    X_labeled_augment_mcar = [X_labeled_augment_mcar; this_x_unlabeled]; 
    index = X_unlabeled_remain_mcar ~= this_x_unlabeled;
    X_unlabeled_remain_mcar = X_unlabeled_remain_mcar(index(:, 1), :);
    y_unlabeled_remain_mcar = y_unlabeled_remain_mcar(index(:, 1));
    p_y_1_x = pdf(p_xy_1, this_x_unlabeled)/(pdf(p_xy_1, this_x_unlabeled) + pdf(p_xy_0, this_x_unlabeled));
    y_labeled_augment_mcar = [y_labeled_augment_mcar; binornd(1, p_y_1_x)];
        
end
toc

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

% % p'(x)
p_x_new = @(x1, x2) (p_y_0_new*pdf(p_xy_0_new, [x1 x2]) + p_y_1_new*pdf(p_xy_1_new, [x1 x2]) + p_y_unlabeled_new*pdf(p_xy_unlabeled_new, [x1 x2])); 

% Test
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

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
scatter(X_unlabeled(:, 1), X_unlabeled(:, 2), 50,...
    p_y0_x, 'filled', 'o', 'LineWidth', 3, 'MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5);
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

%% New probabilities (MCAR)
K = 10; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% p'(x | y = 1)
y_1 = y_labeled_augment_mcar(y_labeled_augment_mcar == 1); 
X_1 = X_labeled_augment_mcar(y_labeled_augment_mcar == 1, :); 
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

% p'(x | y = 0)
y_0 = y_labeled_augment_mcar(y_labeled_augment_mcar == 0);
X_0 = X_labeled_augment_mcar(y_labeled_augment_mcar == 0, :);
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

% % p'(x)
p_x_new = @(x1, x2) (p_y_0_new*pdf(p_xy_0_new, [x1 x2]) + p_y_1_new*pdf(p_xy_1_new, [x1 x2])); 

% Test
tic
y_uncertainty = zeros(N_unlabeled, 1);
y_predict= zeros(N_unlabeled, 1); 
p_y0_x = zeros(N_unlabeled, 1); 
for n = 1:N_unlabeled
    
        this_x_unlabeled = X_unlabeled(n, :);
        this_p_yx_0 = (pdf(p_xy_0_new, this_x_unlabeled)*p_y_0_new)/p_x_new(this_x_unlabeled(1), this_x_unlabeled(2));
        this_p_yx_1 = (pdf(p_xy_1_new, this_x_unlabeled)*p_y_1_new)/p_x_new(this_x_unlabeled(1), this_x_unlabeled(2));
        p_y0_x(n) = this_p_yx_0;

end
toc

figure('position', [100, 100, 600, 600]); % Scatter plot 
hold on;
scatter(X_unlabeled(:, 1), X_unlabeled(:, 2), 50,...
    p_y0_x, 'filled', 'o', 'LineWidth', 3, 'MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5);
xlim([-5 5]);
ylim([-5 5]);
colormap(jet)
colorbar;
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 20, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'p_y0_x_mcar.png'));
saveas(gcf, fullfile(fpath, 'p_y0_x_mcar.fig'));

%% The Example

% Figure
figure('position', [100, 100, 600, 600]);
hold on
scatter(X_unlabeled((y_unlabeled==0), 1), X_unlabeled((y_unlabeled==0), 2), 100, 'o', 'LineWidth', 3,...
    'MarkerFaceColor','w','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
scatter(X_unlabeled((y_unlabeled==1), 1), X_unlabeled((y_unlabeled==1), 2), 100, 'x', 'LineWidth', 3,...
    'MarkerFaceColor','k','MarkerEdgeColor','k','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
scatter(X_labeled((y_labeled==0),1), X_labeled((y_labeled==0),2), 100, 'or', 'LineWidth', 3);
scatter(X_labeled((y_labeled==1),1), X_labeled((y_labeled==1),2), 100, 'xb', 'LineWidth', 3);

xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 20, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'data.png'));
saveas(gcf, fullfile(fpath, 'data.fig'));