clear all
fclose all 

%% All data 0-9
number = zeros(28, 28, 1000, 10);
for n = 1:10
    filename = ['data' num2str(n-1)];
    fid=fopen(filename, 'r'); % open the file corresponding to digit 8
    for i = 1:1000
    number(:, :, i, n) = fread(fid, [28 28], 'uchar');
    end
end

%% p(x | y), y = 0, ..., 9
mu_xy = zeros(28, 28, 10);
for n = 1:10
    mu_xy(:, :, n) = sum(number(:, :, :, n), 3)/1000;
end

imagesc(mu_xy(:, :, 1)');

sigma_xy = zeros(28^2, 28^2, 10);
for n = 1:10
    this_mu = mu_xy(:, :, n); 
    this_mu = reshape(this_mu, 28^2, 1);
    tic
    for i = 1:1000
        this_number = number(:, :, i, n);
        this_number = reshape(this_number, 28^2, 1);
        
        sigma_xy(:, :, n) = sigma_xy(:, :, n) + (this_number - this_mu)*(this_number - this_mu)';
    end
    sigma_xy(:, :, n) = sigma_xy(:, :, n)/1000;
    toc
end
 
% imagesc(sigma_xy(:, :, 1)');

%% p(x) 

mu_x = sum(mu_xy, 3)/10;
% imagesc(mu_x(:, :)');

sigma_x = sum(sigma_xy, 3)/10;
% imagesc(sigma_x(:, :)');

%% p(y) = 1/10


