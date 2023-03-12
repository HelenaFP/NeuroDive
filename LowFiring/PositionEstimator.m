function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
  
  % remove low firing neurons from test_data
  low_fire_n = modelParameters.low_fire_n;
  test_data.spikes(low_fire_n,:) = [];

  names = ["angle1" "angle2" "angle3" "angle4" "angle5" "angle6" "angle7" "angle8"];
  if ~isempty(test_data.decodedHandPos)
      reaching_angle = modelParameters.reaching_angle;
  else
      reaching_angle = lda_classifier(test_data, modelParameters);
      modelParameters.reaching_angle = reaching_angle;
  end
  
  pca = modelParameters.(names(reaching_angle)).pca;
  lre = modelParameters.(names(reaching_angle)).lre;
    
  end_spikes = size(test_data.spikes, 2);
  spikes_300ms = test_data.spikes(:, end_spikes-249:end_spikes);
  spikes_300ms = (spikes_300ms' - pca.mean)';
  spikes_300ms = reshape(pca.loadings'*spikes_300ms,[],1);

  % Adding a column of ones to X to account for the intercept term 
  % i.e. y = b0*1 + b1*x1 + b2*x2 +...
  X = [1; spikes_300ms]';
  hand_position = X * lre;
  x = hand_position(1);
  y = hand_position(2);
   
end

function [reaching_angle] = lda_classifier(data, modelParameters)
    
    rate = mean(data.spikes(:, 1:320), 2)';
    scores = rate * modelParameters.lda_coef' + modelParameters.lda_intercept;
    [~, reaching_angle] = max(scores, [], 2);
    
end
  


