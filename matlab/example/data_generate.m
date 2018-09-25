%% Generate random inputs with a Gaussian mixture model
%% x
mu_x = [0 2; 0 -2];    % Mean vector 
COV_x = cat(3, [1 0; 0 .5], [1 0; 0 .5]);   % Cov matrix
p = ones(1, 2)/2;       % Weights of components       
gm = gmdistribution(mu_x, COV_x, p);    % GMM model for generating inputs

% X = random(gm, N_labeled + N_unlabeled + N_testing);  
if N_unlabeled ~= 0
X = random(gm, N_labeled + N_unlabeled); 
else
X = random(gm, 300);     
end

% Quadrant
X_Q1 = X(((X(:, 1)>=0) & (X(:, 2)>=0)), :);
X_Q2 = X(((X(:, 1)<0) & (X(:, 2)>=0)), :);
X_Q3 = X(((X(:, 1)<0) & (X(:, 2)<0)), :);
X_Q4 = X(((X(:, 1)>=0) & (X(:, 2)<0)), :);

%% y
y = zeros(length(X), 1);
for n = 1:length(X)
    if X(n, 1) < 0
        y(n) = 1;
    else 
        y(n) = 0;
    end
end

y_Q1 = y(((X(:, 1)>=0) & (X(:, 2)>=0)));
y_Q2 = y(((X(:, 1)<0) & (X(:, 2)>=0)));
y_Q3 = y(((X(:, 1)<0) & (X(:, 2)<0)));
y_Q4 = y(((X(:, 1)>=0) & (X(:, 2)<0)));
   
%% Labeled and unlabeled
X_labeled = [X_Q1(1:N_labeled/2, :); X_Q3(1:N_labeled/2, :)];
y_labeled = [y_Q1(1:N_labeled/2); y_Q3(1:N_labeled/2)];

if N_unlabeled  ~= 0
X_unlabeled = [X_Q1(N_labeled/2+1:end, :); X_Q2; X_Q3(N_labeled/2+1:end, :); X_Q4];
y_unlabeled = [y_Q1(N_labeled/2+1:end); y_Q2; y_Q3(N_labeled/2+1:end); y_Q4];
else
y_unlabeled = [];    
end  

% X_labeled = X(1:N_labeled, :);
% X_unlabeled = X(N_labeled+1:N_labeled+N_unlabeled, :);
% X_testing = X(N_labeled+N_unlabeled+1:N_labeled+N_unlabeled+N_testing, :);
% 
% y_labeled = y(1:N_labeled);
% y_unlabeled = y(N_labeled+1:N_labeled+N_unlabeled);
% y_testing = y(N_labeled+N_unlabeled+1:N_labeled+N_unlabeled+N_testing);

%% Visualize
if N_unlabeled  ~= 0
figure('position', [100, 100, 600, 600]);
hold on
scatter(X_labeled((y_labeled==0),1), X_labeled((y_labeled==0),2), 100, 'or', 'LineWidth', 3);
scatter(X_labeled((y_labeled==1),1), X_labeled((y_labeled==1),2), 100, 'xb', 'LineWidth', 3);
scatter(X_unlabeled((y_unlabeled==0), 1), X_unlabeled((y_unlabeled==0), 2), 50, 's', 'LineWidth', 1,...
    'MarkerFaceColor','r','MarkerEdgeColor','r','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
scatter(X_unlabeled((y_unlabeled==1), 1), X_unlabeled((y_unlabeled==1), 2), 50, 's', 'LineWidth', 1,...
    'MarkerFaceColor','b','MarkerEdgeColor','b','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.1);
legend('Labeled, class 0', 'Labeled, class 1', 'Unlabeled, class 0', 'unlabeled, class 1');
% colorbar;
colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'data.png'));
% saveas(gcf, fullfile(fpath, 'data.fig'));

else
figure('position', [100, 100, 600, 600]);
hold on
scatter(X_labeled((y_labeled==0),1), X_labeled((y_labeled==0),2), 100, 'or', 'LineWidth', 3);
scatter(X_labeled((y_labeled==1),1), X_labeled((y_labeled==1),2), 100, 'xb', 'LineWidth', 3);
legend('Labeled, class 0', 'Labeled, class 1');
% colorbar;
colormap(jet);
xlim([-5 5]);
ylim([-5 5]);
caxis([0 1]);
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontWeight', 'bold')
% saveas(gcf, fullfile(fpath, 'data.png'));
% saveas(gcf, fullfile(fpath, 'data.fig'));
end