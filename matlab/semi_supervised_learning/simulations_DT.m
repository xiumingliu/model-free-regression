disp('Simulation decision test start ... ')

[X1_test_DT, X2_test_DT] = meshgrid(-5:.1:5);
N_testing_DT = length(X1_test_DT);
y_predict_sim = zeros(N_testing_DT, N_testing_DT, NUM_SIM_DT); 


for num_sim = 1:NUM_SIM_DT
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

%%

for row = 1:N_testing_DT
    for col = 1:N_testing_DT
        this_x_testing = [X1_test_DT(row,col), X2_test_DT(row,col)];
        if (pdf(this_p_xy_0, this_x_testing)*p_y_0) >= (pdf(this_p_xy_1, this_x_testing)*p_y_1)
            y_predict_sim(row, col, num_sim) = 0;
        else
            y_predict_sim(row, col, num_sim) = 1;
        end 
    end
end

disp(num_sim)
toc
end

y_predict_dif = zeros(N_testing_DT, N_testing_DT); 
y_predict_map = zeros(N_testing_DT, N_testing_DT); 
for row = 1:N_testing_DT
    for col = 1:N_testing_DT
        this_x_testing = [X1_test_DT(row,col), X2_test_DT(row,col)];
        if (pdf(p_xy_0, this_x_testing)*p_y_0) >= (pdf(p_xy_1, this_x_testing)*p_y_1)
            y_predict_map(row, col) = 0;
        else
            y_predict_map(row, col) = 1;
        end 
        
        y_predict_dif(row, col) = sum(y_predict_map(row, col) ~= y_predict_sim(row, col, :));
        
    end
end

y_predict_dif = y_predict_dif/NUM_SIM_DT;

figure('position', [100, 100, 600, 600]);
hold on;
% imagesc(y_predict_dif);
contourf(X1_test_DT, X2_test_DT, y_predict_dif);
colormap(jet)
colorbar;
xlim([-5 5]);
ylim([-5 5]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'simulation_decision.png'));
% saveas(gcf, fullfile(fpath, 'simulation_decision.fig'));