function [test] = knn_angle(k,train,test)

%angles_list = [pi/6,7*pi/18,11*pi/18,5*pi/6,19*pi/18,23*pi/18,31*pi/18,35*pi/18];

[trials_per_angle_train, nangles] = size(train);
[trials_per_angle_test, ~] = size(train);


% Spike count array. Each column corresponds to spike counts from one
% trial. Each column corresponds to a certain angle
spikes_train = [];
spikes_test = [];


for a=1:nangles
   for i=1:trials_per_angle_train
       spikes_train = [spikes_train, sum(train(i,a).spikes(:,1:320),2)];
   end
   for i=1:trials_per_angle_test
       spikes_test = [spikes_test, sum(test(i,a).spikes(:,1:320),2)];
   end
end


ntrials = trials_per_angle_test*nangles;


for tr = 1:ntrials
    
    tr_spikes = repmat(spikes_test(:,tr),1,ntrials);
    
    
    % Get the distances between all of the trained vectors and the current
    % test vector
    
    dists = vecnorm((spikes_train-tr_spikes),2,1);
    
    [~, minInds] = mink(dists,k);
    
    test(tr).angle_class = mode([train(minInds).angle_id]);
    
    test(tr).ang_correct = test(tr).angle_class == test(tr).angle_id;
    




end

