%% Generate random inputs with a Gaussian mixture model

mu_x = [0 0; 0 0];    % Mean vector 
COV_x = cat(3, [1 0; 0 1], [1 0; 0 1]);   % Cov matrix
p = ones(1, 2)/2;       % Weights of components       
gm = gmdistribution(mu_x, COV_x, p);    % GMM model for generating inputs

X = random(gm, N_labeled + N_unlabeled + N_testing);    

X_labeled = X(1:N_labeled, :);
X_unlabeled = X(N_labeled+1:N_labeled+N_unlabeled, :);
X_testing = X(N_labeled+N_unlabeled+1:N_labeled+N_unlabeled+N_testing, :);

%% Generate the outputs from a logistic regression model
% Construct the GP: g ~ GP
% alpha = 1; theta = 1;
% f_mean = @(x) zeros(length(x), 1);
% f_cov = @(x1, x2) alpha^2*exp(-(norm(x1 - x2))^2/(2*theta^2));
% 
% mu_g = f_mean(X);
% COV_g = zeros(N_labeled + N_unlabeled + N_testing, N_labeled + N_unlabeled + N_testing);
% for row = 1:N_labeled + N_unlabeled + N_testing
%     for col = 1:N_labeled + N_unlabeled + N_testing
%         COV_g(row, col) = f_cov(X(row, :), X(col, :));
%     end
% end
% g = mvnrnd(mu_g, COV_g)';
% 
% % z = g + epsilon
% var_epsilon = .01; 
% epsilon = normrnd(0, sqrt(var_epsilon), N_labeled + N_unlabeled + N_testing, 1);
% z = g + epsilon; 
% 
% % Generate outputs y
% y = round(logsig(z));

y = zeros((N_labeled + N_unlabeled + N_testing), 1);
for n = 1:(N_labeled + N_unlabeled + N_testing)
    if sqrt(X(n, 1)^2 + X(n, 2)^2) < 1
        y(n) = 1;
    else 
        y(n) = 0;
    end
end
    
y_labeled = y(1:N_labeled);
y_unlabeled = y(N_labeled+1:N_labeled+N_unlabeled);
y_testing = y(N_labeled+N_unlabeled+1:N_labeled+N_unlabeled+N_testing);