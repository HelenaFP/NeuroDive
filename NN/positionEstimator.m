function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)


  if size(test_data.spikes, 2) == 320
      x = test_data.startHandPos(1);
      y = test_data.startHandPos(2);


  else
      % remove low firing neurons from test_data
      low_fire_n = modelParameters.low_fire_n;
      test_data.spikes(low_fire_n,:) = [];

      n_neurons = size(test_data.spikes,1);


      % Find angle

      spike_sums_320 = sum(test_data.spikes(:,1:320),2);

      angle_class_net = modelParameters.angle_class_net;


      angle_probabilities = angle_class_net(spike_sums_320);

      [~,angle] = max(angle_probabilities);



      % Get requred network object for that angle

      names = ["angle1" "angle2" "angle3" "angle4" "angle5" "angle6" "angle7" "angle8"];


      netr = modelParameters.(names(angle));


      % Get all spike data from 320 onwards

      bin_width = 20;

      t_length = size(test_data.spikes,2);

      edges = 320:bin_width:t_length;

      bins = diff(edges);


      spikes_after_320 = [];
      

      % binned spike counts for each neuron
      for neuron = 1:n_neurons
          spike_inds = find(test_data.spikes(neuron,:));
          [binned_spikes,~] = histcounts(spike_inds,edges);
          spikes_after_320 = [spikes_after_320;binned_spikes];

      end

      spikes_after_320 = spikes_after_320./bins;

      net_positions = netr(spikes_after_320);

      positions = [test_data.startHandPos(1);test_data.startHandPos(2)];
      positions = [positions net_positions];

      x_interpolate = interp1(edges,positions(1,:),320:size(test_data.spikes,2),'linear');
      y_interpolate = interp1(edges,positions(2,:),320:size(test_data.spikes,2),'linear');

      

      x = x_interpolate(end);
      y = y_interpolate(end);













      
  end







    
   
end