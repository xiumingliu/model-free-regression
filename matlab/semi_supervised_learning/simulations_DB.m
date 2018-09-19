for num_sim = 1:20
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

this_p_xy0 = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

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

this_p_xy1 = gmdistribution(this_mu_hat', this_COV_hat, this_pi_hat);

fcontour(@(x1, x2)(pdf(this_p_xy1, [x1, x2])*p_y_1 - pdf(this_p_xy0, [x1, x2])*p_y_0),...
    [-5 5 -5 5], '--b', 'LevelList', [0], 'LineWidth', 1);

end

figure('position', [100, 100, 600, 600]);
hold on;
scatter(X_testing((y_testing==0),1), X_testing((y_testing==0),2), 50, 'ok');
scatter(X_testing((y_testing==1),1), X_testing((y_testing==1),2), 50, 'xk');
scatter(X_testing(find(y_testing - y_predict), 1), X_testing(find(y_testing - y_predict), 2), 100, 'square', 'r', 'filled'); 
xlim([-5 5]);
ylim([-5 5]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
saveas(gcf, fullfile(fpath, 'simulation_decision.png'));
saveas(gcf, fullfile(fpath, 'simulation_decision.fig'));