function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)

names = ["angle1" "angle2" "angle3" "angle4" "angle5" "angle6" "angle7" "angle8"];
    [trials_per_angle, nangles] = size(test_data);
    ntrials = trials_per_angle* nangles;

    %%%%%%%%%%%%%%%%%%%%%%%% PUT IN NEW BINS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    test_new_bins = struct;

    new_bin = 10; % ms

    all_spikes = test_data.spikes; % spikes is no neurons x no time points
    nNeurons = size(all_spikes,1);
    t_length = size(all_spikes,2);
    t_new = 1: new_bin : t_length; % New time point array
    spikes = zeros(nNeurons,numel(t_new)-1);

    for k = 1 : numel(t_new)-1 % get rid of the paddded bin
        spikes(:,k) = sum(all_spikes(:,t_new(k):t_new(k+1)-1),2);
    end

    spikes = sqrt(spikes);
    %binned_handPos = handPos(:,t_new);

    test_new_bins.spikes = spikes;
            %test_new_bins(j,i).handPos = binned_handPos;


    %%%%%%%%%%%%%%%%%%%%%%%%% FIRING RATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    % Using a gaussian kernel

    test_rates = struct;

    scale_win = 50;

    std = scale_win/new_bin;
    w = 10*std;

    alpha = (w-1)/(2*std);
    temp1 = -(w-1)/2 : (w-1)/2;
    gausstemp = exp((-1/2) * (alpha * temp1/((w-1)/2)) .^ 2)';
    gaussian_window = gausstemp/sum(gausstemp);

    conv_rates = zeros(size(test_new_bins.spikes,1),size(test_new_bins.spikes,2));

    for k = 1: size(test_new_bins.spikes,1)

        % COnvolve with gaussian window
        conv_rates(k,:) = conv(test_new_bins.spikes(k,:),gaussian_window,'same')/(new_bin/1000);
    end

    test_rates.rates = conv_rates;
    test_rates.spikes = test_new_bins.spikes;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%% REMOVE LOW FIRING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    % find and remove low firing neurons from training data, and store
    % these neurons (indices) to be removed from test data

  % remove low firing neurons from test_data
  low_fire_n = modelParameters.low_fire_n;
  test_rates.spikes(low_fire_n,:) = [];
  test_rates.rates(low_fire_n,:) = [];
  
  nNeurons = size(test_rates.rates,1);

 
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%% ANGLE CLASSIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
  

  if ~isempty(test_data.decodedHandPos)
      reaching_angle = modelParameters.reaching_angle;
  else
      Wtrain = modelParameters.ldaW;
      %nPCA = modelParameters.lda_npca;
      opt_proj = modelParameters.lda_opt_proj;
      nLDA = modelParameters.lda_nlda;
      %meanFiringTrain = modelParameters.classify(indexer).mFire_kNN;
      trimmer = 320/new_bin;
      
      fire_rates = reshape(test_rates.rates, [], 1);
      
      Wtest = opt_proj'*(fire_rates-mean(fire_rates)); 
      
      k = 20;
      [outLabel] = knn_angles(k,Wtrain,Wtest);
      

      modelParameters.actualLabel = outLabel;
      if outLabel ~= modelParameters.actualLabel
          outLabel = modelParameters.actualLabel;

      end
  end
  
  pca = modelParameters.(names(outLabel)).pca;
  lre = modelParameters.(names(outLabel)).lre;
    
  end_spikes = size(test_data.spikes, 2);
  spikes_300ms = test_data.spikes(:, end_spikes-249:end_spikes);
  spikes_300ms = (spikes_300ms' - pca.mean)';
  spikes_300ms = reshape(pca.loadings'*spikes_300ms,[],1);

  % Adding a column of ones to X to account for the intercept term 
  % i.e. y = b0*1 + b1*x1 + b2*x2 +...
  fire_rates = [1; spikes_300ms]';
  hand_position = fire_rates * lre;
  x = hand_position(1);
  y = hand_position(2);
   
end

function [reaching_angle] = lda_classifier(data, modelParameters)
    
    rate = mean(data.spikes(:, 1:320), 2)';
    scores = rate * modelParameters.lda_coef' + modelParameters.lda_intercept;
    [~, reaching_angle] = max(scores, [], 2);
    
end


function [angle_label] = knn_angles(k,Wtrain,Wtest)

ntrials = size(Wtrain,2);
nangles_trial = ntrials/ 8;

all_labels = [1*ones(1,nangles_trial),2*ones(1,nangles_trial),3*ones(1,nangles_trial),4*ones(1,nangles_trial),5*ones(1,nangles_trial),6*ones(1,nangles_trial),7*ones(1,nangles_trial),8*ones(1,nangles_trial)]';

test_mat = repmat(Wtest,1,ntrials);

dists = vecnorm(Wtrain-test_mat);

[~,idx] = sort(dists);

nn_inds = idx(1:k);

p_labs = all_labels(nn_inds);

angle_label = mode(p_labs);
    


    
end


