function [modelParameters] = positionEstimatorTraining(training_data)
    modelParameters = struct;
    
    % find and remove low firing neurons from training data, and store
    % these neurons (indices) to be removed from test data
    low_fire_n = get_low_firing_neurons(training_data);
    modelParameters.low_fire_n = low_fire_n;
    for n = 1:length(training_data)
        for k = 1:8
            training_data(n,k).spikes(low_fire_n,:) = [];
        end
    end
     
    n_neurons = size(training_data(1,1).spikes,1);
     
     
    names = ["angle1" "angle2" "angle3" "angle4" "angle5" "angle6" "angle7" "angle8"];
     
    trials_per_angle = size(training_data,1);
     
     
    % First neural network classifier to determine angle
         
    % Angle decided in the first 320 ms
    
    spike_sums_320 = zeros(n_neurons,trials_per_angle*8);
    labels = zeros(trials_per_angle*8,1);
     
     
    i = 1;
    for angle = 1:8
     
        for tr = 1:trials_per_angle
    
            spike_sums_320(:,i) = sum(training_data(tr,angle).spikes(:,1:320),2);
            labels(i) = angle;
            i = i + 1;
    
        end
    end
    

    target = ind2vec((labels).');
    netc = patternnet(5,'trainlm');
    netc.divideParam.trainRatio = 80/100;
    netc.divideParam.valRatio = 20/100;
    netc.divideParam.testRatio = 0/100;
    netc = train(netc,spike_sums_320,target);
     
    modelParameters.angle_class_net = netc;

    % For each angle, a network for linear regression to approximate the
    % trajectory

    for angle = 1:8


        % Get the sum of spikes after 320 ms if it is there. If there is
        % only 320ms there, the initially given handpos is returned

        spikes_after_320 = [];
        positions = [];


        % Bin in 20 ms bins
        bin_width = 20;

        for tr = 1:trials_per_angle

            t_length = size(training_data(tr,angle).spikes,2);

            if t_length > 320

                spikes_neuron = [];

                edges = 320:bin_width:t_length;

                if edges(length(edges)) ~= t_length
                    edges(length(edges)+1) = t_length;
                end

                bins = diff(edges);



                % binned spike counts for each neuron
                for neuron = 1:n_neurons

                    spike_inds = find(training_data(tr,angle).spikes(neuron,:));
                    [binned_spikes,~] = histcounts(spike_inds,edges);
                    spikes_neuron = [spikes_neuron;binned_spikes];

                end

                avg_spike_neuron = spikes_neuron./bins;

                spikes_after_320 = [spikes_after_320 avg_spike_neuron];
                
                all_pos = training_data(tr,angle).handPos(1:2,:);

                avg_pos = zeros(2,length(edges)-1);

                for edge=1:length(edges)-1

                    avg_pos(:,edge) = mean(all_pos(1:2,edges(edge):edges(edge+1)-1),2);

                end

                    






                positions = [positions avg_pos];
            end
        end

        %pos_target = ind2vec((positions).');

        netr = fitnet(10,'trainlm');
        netr.divideParam.trainRatio = 80/100;
        netr.divideParam.valRatio = 20/100;
        netr.divideParam.testRatio = 0/100;

        % Train NN
        [netr, ~] = train(netr, spikes_after_320, positions);  


        modelParameters.(names(angle)) = netr;




    end






        















































end


function low_fire_n = get_low_firing_neurons(training_data)
    firing_rate = [];
    firing_rate_angle = [];
    for k = 1:8 
        for n = 1:length(training_data)
            firing_rate = [firing_rate mean(training_data(n,k).spikes(:,:),2)];
        end
        firing_rate_angle =[firing_rate_angle mean(firing_rate,2)];
    end
    average_firing_rate = mean(firing_rate_angle,2); 
    var_firing_rate = std(firing_rate_angle,0,2);

    % find the n neurons with the lowest average firing rate 
    [low_rates,low_avg_n] = mink(average_firing_rate,5);
    % find the n neurons with the lowest varying firing rate
    [low_var, low_var_n] = mink(var_firing_rate,5);
    % find m neurons which meet both conditions, where m<=n
    low_fire_n = intersect(low_avg_n,low_var_n);
    display(low_fire_n);
end