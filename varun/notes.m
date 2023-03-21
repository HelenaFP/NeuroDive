
clear
clc



load('monkeydata_training.mat');


% Number of trials = 100
% Reaching angles = 8

% 100 trials for each reaching angle

% for all 100*8 experiments:  there are 98 neurons each with a spike train
% and x,y data



% Need to find a mapping between the spike train and the x, Y data
%[anglenet] = angle_trainer_nn(trial);


[trials_per_angle, nangles] = size(trial);


for j=1:nangles
    for i=1:trials_per_angle
        trial(i,j).angle_id = j;
    end
end

ix = randperm(length(trial));
% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

k = 10;
test = knn_angle(k,trainingData,testData);


correct = [test.ang_correct];

accuracy = sum(correct)/(size(test,1)*size(test,2));



x = 0;




% COnvolution filter on the spike trains
% Can use multiple convolutive filters in series.

% Get instantaneous velocity
% Dimensionality reduction, spike trains


% Linear regression??? Between spike and velocity rates 
% Least mean square filter - start with this

% Simple neural network architecture 
% Spiking neural nets??? 


% Firing rate = number of spikes per time bin / time bin

% Normalized hamming or hanning filter

% Spike counts more relevant for the position 

















% 
