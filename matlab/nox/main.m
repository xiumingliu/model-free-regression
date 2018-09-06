clear all 
close all

load('data.mat');
load('no2.mat');


weekday_hour = [weekday(time) time.Hour];
figure;
scatter3(weekday_hour(:,1), weekday_hour(:,2), no2(:, 1), 'filled')
figure;
no2_3 = no2(:, 1);
no2_3 = no2_3(weekday(time) == 3);
scatter3(temperature(weekday(time) == 3), windSpeed(weekday(time) == 3), no2_3, 'filled')

no2((no2(:, 4) > 200), 4);