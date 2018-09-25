[X1_test_EP, X2_test_EP] = meshgrid(-5:.25:5);
N_testing_EP = length(X1_test_EP);
y_predict_EP_sim = zeros(N_testing_EP, N_testing_EP, NUM_SIM_EP); 

disp('Simulation error probability start ... ')
for num_sim = 1:NUM_SIM_EP
tic
% For p(x | y = 0)
N_k = sum(model_xy_0.R);

this_mu = (model_xy_0.m)';
this_cov = zeros(D, D, K);
this_W = model_xy_0.W;
for k = 1:K
    this_W(2, 1, k) = this_W(1, 2, k);
    this_cov(:, :, k) = (model_xy_0.beta(k)*this_W(:, :, k))^-1;
    [~, isspd] = chol(this_cov(:, :, k));
    multiplier = 1;
    while isspd
        this_cov(:, :, k) = this_cov(:, :, k) + (0.01*10^(multiplier))*eye(D, D);
        multiplier = multiplier + 1;
        [~, isspd] = chol(this_cov(:, :, k));
    end 
end
this_mu_hat = mvnrnd(this_mu, this_cov)';
this_COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_xy_0.R(:, k);
    this_COV_hat(:, :, k) = (1/N_k(k))*((sqrt(r_nk').*(X_0' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X_0' - this_mu_hat(:, k)))');
end
this_pi_hat = (model_xy_0.alpha/sum(model_xy_0.alpha));

this_p_xy_0 = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

% For p(x | y = 1)
N_k = sum(model_xy_1.R);

this_mu = (model_xy_1.m)';
this_cov = zeros(D, D, K);
this_W = model_xy_1.W;
for k = 1:K
    this_W(2, 1, k) = this_W(1, 2, k);
    this_cov(:, :, k) = (model_xy_1.beta(k)*this_W(:, :, k))^-1;
    [~, isspd] = chol(this_cov(:, :, k));
    multiplier = 1;
    while isspd
        this_cov(:, :, k) = this_cov(:, :, k) + (0.01*10^(multiplier))*eye(D, D);
        multiplier = multiplier + 1;
        [~, isspd] = chol(this_cov(:, :, k));
    end 
end
this_mu_hat = mvnrnd(this_mu, this_cov)';
this_COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_xy_1.R(:, k);
    this_COV_hat(:, :, k) = (1/N_k(k))*((sqrt(r_nk').*(X_1' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X_1' - this_mu_hat(:, k)))');
end
this_pi_hat = (model_xy_1.alpha/sum(model_xy_1.alpha));

this_p_xy_1 = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

% For p(x | y = u)
if N_unlabeled ~= 0
N_k = sum(model_xy_unlabled.R);

this_mu = (model_xy_unlabled.m)';
this_cov = zeros(D, D, K);
this_W = model_xy_unlabled.W;
for k = 1:K
    this_W(2, 1, k) = this_W(1, 2, k);
    this_cov(:, :, k) = (model_xy_unlabled.beta(k)*this_W(:, :, k))^-1;
    [~, isspd] = chol(this_cov(:, :, k));
    multiplier = 1;
    while isspd
        this_cov(:, :, k) = this_cov(:, :, k) + (0.01*10^(multiplier))*eye(D, D);
        multiplier = multiplier + 1;
        [~, isspd] = chol(this_cov(:, :, k));
    end 
end
this_mu_hat = mvnrnd(this_mu, this_cov)';
this_COV_hat = zeros(D, D, K);
for k = 1:K
    r_nk = model_xy_unlabled.R(:, k);
    this_COV_hat(:, :, k) = (1/N_k(k))*((sqrt(r_nk').*(X_unlabeled' - this_mu_hat(:, k)))*(sqrt(r_nk').*(X_unlabeled' - this_mu_hat(:, k)))');
end
this_pi_hat = (model_xy_unlabled.alpha/sum(model_xy_unlabled.alpha));

this_p_xy_unlabeled = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);
end

% Marginal distribution p(x)
if N_unlabeled ~= 0
this_p_x = @(x1, x2) (p_y_0*pdf(this_p_xy_0, [x1 x2]) + p_y_1*pdf(this_p_xy_1, [x1 x2])...
    + p_y_unlabeled*pdf(this_p_xy_unlabeled, [x1 x2])); 
else
this_p_x = @(x1, x2) (p_y_0*pdf(this_p_xy_0, [x1 x2]) + p_y_1*pdf(this_p_xy_1, [x1 x2]));
end
%% Error probability
for row = 1:N_testing_EP
    for col = 1:N_testing_EP
        this_x_testing = [X1_test_EP(row,col), X2_test_EP(row,col)];
        % Normalization
        this_p_yx_0 = (pdf(this_p_xy_0, this_x_testing)*p_y_0)/this_p_x(this_x_testing(1), this_x_testing(2));
        this_p_yx_1 = (pdf(this_p_xy_1, this_x_testing)*p_y_1)/this_p_x(this_x_testing(1), this_x_testing(2));
%         this_normalizor = 1/(this_p_yx_0 + this_p_yx_1);
%         this_p_yx_0 = this_p_yx_0*this_normalizor;
%         this_p_yx_1 = this_p_yx_1*this_normalizor;

        if this_p_yx_0 >= this_p_yx_1
            % Decision 0
            y_predict_EP_sim(row, col, num_sim) = 1 - this_p_yx_0;
        else
            % Decision 1
            y_predict_EP_sim(row, col, num_sim) = 1 - this_p_yx_1;
        end
    end
end
disp(num_sim)

toc
end

var_EP = var(y_predict_EP_sim, 0, 3);
confidence_EP = 2*std(y_predict_EP_sim, 0, 3);

figure('position', [100, 100, 600, 600]);
hold on;
contourf(X1_test_EP, X2_test_EP, var_EP);
colormap(jet)
colorbar;
xlim([-5 5]);
ylim([-5 5]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'simulation_error_var.png'));
saveas(gcf, fullfile(fpath, 'simulation_error_var.fig'));

figure('position', [100, 100, 600, 600]);
hold on;
contourf(X1_test_EP, X2_test_EP, confidence_EP);
colormap(jet)
colorbar;
xlim([-5 5]);
ylim([-5 5]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'simulation_error_2std.png'));
saveas(gcf, fullfile(fpath, 'simulation_error_2std.fig'));