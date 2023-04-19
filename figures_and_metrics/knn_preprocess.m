train_new_bins = struct;
new_bin = 10; % ms
for i = 1:nangles
    for j = 1:trials_per_angle

        all_spikes = training_data(j,i).spikes; % spikes is no neurons x no time points
        nneurons = size(all_spikes,1);
        t_length = size(all_spikes,2);
        t_new = 1: new_bin : t_length; % New time point array
        handPos = training_data(j,i).handPos(1:2,:);
        spikes = zeros(nneurons,numel(t_new)-1);

        for k = 1 : numel(t_new) - 1 % get rid of the paddded bin
            spikes(:,k) = sum(all_spikes(:,t_new(k):t_new(k+1)-1),2);
        end

        spikes = sqrt(spikes);
        binned_handPos = handPos(:,t_new);

        train_new_bins(j,i).spikes = spikes;
        train_new_bins(j,i).handPos = binned_handPos(:,1:length(binned_handPos)-1);
    end
end   
   
% Using a gaussian kernel
[train_rates] = gaussian(train_new_bins,new_bin);

ntrials = size(training_data, 1);
n_timepoints = 320/new_bin - 1; % make the trajectories the same length

[modelParameters] = knn_train(modelParameters,train_rates,n_timepoints);
modelParameters.n_timepoints= n_timepoints ;