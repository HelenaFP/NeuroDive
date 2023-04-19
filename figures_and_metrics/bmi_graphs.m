% BMI Challenge - Graphs for measurements
% NeuroDive
% Calista Adele Yapeter
% 01487343

%% PCA components (from bmi_training_results Sheet 4 and 5)
% old vs new velocity estimator
% neurons removed = 0

rmse_pca = [14.9125 14.8558 14.8678 14.6859 14.5770 14.5758 14.6640 14.7201];
t_pca = [28.8552 28.1706 27.0385 26.0549 24.0779 23.5175 23.3561 22.3826];
rmse_pca_old = [15.1535 15.1113 15.1167 14.9628 14.9844 14.9446 15.0716 15.1357];
t_pca_old = [28.8896 26.509 26.3369 24.5295 22.0625 22.8252 22.0747 21.2621];
pca = [22 20 18 16 14 13 12 11];

figure(1);
yyaxis left
plot(pca,rmse_pca,'Marker','x','LineWidth',1.5);
hold on;
plot(pca,rmse_pca_old,'Marker','x','LineWidth',1.5);
ylabel('RMSE')
hold on;
yyaxis right
plot(pca,t_pca,'Marker','x','LineWidth',1.5);
hold on;
plot(pca,t_pca_old,'Marker','x','LineWidth',1.5);
legend('New Velocity Estimator with Past Input','Old Velocity Estimator')
xlabel('PCA components')
ylabel('Run time/s')
title('Number of PCA components vs. RMSE and Run Time')
fontsize(gca, 15,'points')
hold off

%% PCA components (from bmi_training_results Sheet 5)
% neurons removed = 0 and 9

rmse_pca_9 = [14.9125 14.8558 14.8678 14.6859 14.5770 14.5758 14.6640 14.7201];
t_pca_9 = [28.8552 28.1706 27.0385 26.0549 24.0779 23.5175 23.3561 22.3826];
rmse_pca_0 = [15.0579 15.0097 15.0238 14.8225 14.7624 14.7578 14.8840 14.8856];
t_pca_0 = [38.5209 36.8318 33.0502 31.6361 28.458 26.4386 25.0799 25.2681];
pca = [22 20 18 16 14 13 12 11];

figure(2);
yyaxis left
plot(pca,rmse_pca_0,'Marker','x','LineWidth',1.5);
hold on;
plot(pca,rmse_pca_9,'Marker','x','LineWidth',1.5);
ylabel('RMSE')
hold on;
yyaxis right
plot(pca,t_pca_0,'Marker','x','LineWidth',1.5);
hold on;
plot(pca,t_pca_9,'Marker','x','LineWidth',1.5);
legend('No neurons removed','9 neurons removed')
xlabel('PCA components')
ylabel('Run time/s')
title('Number of PCA components vs. RMSE and Run Time')
fontsize(gca, 15,'points')
hold off

%% Number of neurons removed (from bmi_training_results Sheet 5)
% PCA components = 13 and 22

rmse_n = [14.7578 15.0219 14.5524 14.5529 14.5546 14.5547 14.5553 14.5563 14.5758 14.5946 14.9323];
t_n = [26.4386 23.8705 24.1892 24.0056 24.1073 23.1781 23.5938 22.4799 22.5737 21.8012 21.849];
neurons = [0 1 2 3 5 6 7 8 9 11 12];

rmse_n_22 = [15.0579 15.3120 14.9041 14.9040 14.9050 14.9158 14.9165 14.9179 14.9125 14.9176 15.2573];
t_n_22 = [38.5209 27.5908 28.3318 27.7566 26.8818 26.7877 26.7766 26.8655 26.7155 26.7851 26.4103];

figure(3);
yyaxis left
plot(neurons,rmse_n,'Marker','x','LineWidth',1.5);
hold on;
plot(neurons,rmse_n_22,'Marker','x','LineWidth',1.5);
hold on;
ylabel('RMSE')

yyaxis right
plot(neurons,t_n,'Marker','x','LineWidth',1.5);
hold on;
plot(neurons,t_n_22,'Marker','x','LineWidth',1.5);
hold on;
legend('PCA Components=13','PCA Components=22')
xlabel('Number of neurons removed')
ylabel('Run time/s')
title('Number of Neurons Removed vs. RMSE and Run Time')
fontsize(gca, 15,'points')
hold off

%% Position estimator T_window (sheet 1)
% win_increment = 15
rmse_win = [16.2989 16.1317 16.0004 16.0147 15.8359 15.7573 15.7498 15.6554 15.6863 15.9818 18.5498];
t_win = [58.2296 57.0980 59.5352 61.4365 64.3738 68.1782 72.2122 73.6257 76.4011 79.6963 90.2002];
t_window = [300 290 280 270 260 250 240 230 220 200 150];

figure(4);
yyaxis left
plot(t_window,rmse_win,'Marker','x','LineWidth',1.5);
ylabel('RMSE')

yyaxis right
plot(t_window,t_win,'Marker','x','LineWidth',1.5);
%legend('RMSE','Run Time')
xlabel('Window Length/samples')
ylabel('Run time/s')
title('Window Length vs. RMSE and Run Time')
fontsize(gca, 15,'points')
hold off