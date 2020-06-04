%% Script for Temporal Data Analysis
% Nick Tacca
% April 15, 2020

clear; close all; clc

%% I. Loading and Sorting Data

% Average arrays
d = [];
rank = [];
score = [];
noError = [];
Error = [];
humanError = [];

% All together arrays
feats = [];
labels = [];

for subject=1:12
    close all;
    sub = num2str(subject);
    if subject < 7
        load(['data/chavarriaga2015_1Dgrid/Subject0' sub '_s1.mat']);
    else
        subject_new = subject-6;
        sub = num2str(subject_new);
        load(['data/chavarriaga2015_1Dgrid/Subject0' sub '_s2.mat'])
    end
    
    % Find indices of labels
    idx_noError = EPO.labels == -1;
    idx_Error = EPO.labels == 1;
    idx_humanError = EPO.labels == 2;
    
    % Sort data
    EPO.noError = EPO.all(:,:,idx_noError);
    EPO.Error = EPO.all(:,:,idx_Error);
    EPO.humanError = EPO.all(:,:,idx_humanError);
    EPO.all(:,:,idx_humanError) = [];
    EPO.labels(idx_humanError) = [];
    
    % Store data
    noError = cat(3, noError, EPO.noError);
    Error = cat(3, Error, EPO.Error);
    humanError = cat(3, humanError, EPO.humanError);

    %% II. Feature Selection

    min_samp = 100;
    max_samp = 500;

    EPO.all_ds = [];
    for i=1:size(EPO.all,1)
        channel_ds = downsample(EPO.all(i,min_samp:max_samp,:),8);
        EPO.all_ds = vertcat(EPO.all_ds,channel_ds);
    end
    
    % Feature vector
    EPO.feat_vec = reshape(EPO.all_ds,[],size(EPO.all_ds,3));
    EPO.feat_vec = EPO.feat_vec';

    % Standardization
    EPO.feat_vec_m = mean(EPO.feat_vec,1);
    EPO.feat_vec_s = std(EPO.feat_vec,0,1);

    % Final feature vector after standardization
    EPO.feat_vec = (EPO.feat_vec - EPO.feat_vec_m)./EPO.feat_vec_s;
    
    % Store data for all subjects and do feature eval for ind subjects
    feats = vertcat(feats, EPO.feat_vec);
    labels = vertcat(labels, EPO.labels);
    [EPO.d, EPO.rank, EPO.score] = fisherrank(EPO.feat_vec,EPO.labels);
    
    % Store feature data for ind subjects
    d = vertcat(d, EPO.d);
    rank = vertcat(rank, EPO.rank);
    score = vertcat(score, EPO.score);
    
end

%% III. Feature Plotting

% All data together
[d_all, rank_all, score_all] = fisherrank(feats,labels);

% Average & Std of Feature Evaluation for all Subjects
score_m = mean(score,1);
score_med = median(score,1);
score_max = max(score,[],1);
score_min = min(score,[],1);
score_v = var(score,0,1);

d_m = mean(d,1);
d_med = median(d,1);
d_max = max(d,[],1);
d_min = min(d,[],1);
d_v = var(d,0,1);
d_up = quantile(d,0.75,1);
d_low = quantile(d,0.25,1);

rank_m = mean(rank,1);
rank_v = var(rank,0,1);

figure
x = 1:numel(d_m);
% curve1 = d_m + d_v;
% curve2 = d_m - d_v;
curve1 = d_up;
curve2 = d_low;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
h1=fill(x2, inBetween, 'b', 'HandleVisibility','off','EdgeColor','None');

% % Saving features
% save("Chav2015_Combo", "d_m", "d_med", "d_all", "d_up", "d_low", "d_min", "d_max", "d_v")

set(h1,'facealpha',0.1)
hold on;
plot(d_m, 'LineWidth',2);
hold on;
plot(d_all, 'LineWidth',2);
title('Chavarriaga 2015 Fisher Rank');
xlabel('Ranked Features')
ylabel('Fisher Score')
ylim([0,1])
grid on
legend('Grand Average Temporal Fisher Scores', 'Overall Temporal Fisher Scores')
% saveas(gcf,'2015_1Dgrid/Combination/FisherRank.png')

%% Log Plots
figure
x = 1:numel(d_m(1:2500));
% curve1 = d_m + d_v;
% curve2 = d_m - d_v;
curve1 = log(d_up(1:2500));
curve2 = log(d_low(1:2500));
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
h1=fill(x2, inBetween, 'b', 'HandleVisibility','off','EdgeColor','None');
set(h1,'facealpha',0.1)
hold on;
plot(log(d_m(1:2500)), 'LineWidth',2);
hold on;
plot(log(d_all(1:2500)), 'LineWidth',2);
xlabel('Ranked Features')
ylabel('Log Fisher Score')
title('Chavarriaga 2015 Log Fisher Rank');
ylim([-8,0])
grid on
legend('Grand Average Temporal Log Fisher Scores', 'Overall Temporal Log Fisher Scores')
% saveas(gcf,'2015_1Dgrid/Combination/LogFisherRank.png')