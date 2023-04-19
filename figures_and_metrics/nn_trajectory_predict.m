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
        netr = fitnet(10,'trainlm');
        netr.divideParam.trainRatio = 80/100;
        netr.divideParam.valRatio = 20/100;
        netr.divideParam.testRatio = 0/100;

        % Train NN
        [netr, ~] = train(netr, spikes_after_320, positions);  
        modelParameters.(names(angle)) = netr;
    end