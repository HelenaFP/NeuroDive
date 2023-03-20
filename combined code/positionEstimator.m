function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
     
  % remove low firing neurons from test_data
  low_fire_n = modelParameters.low_fire_n;
  test_data.spikes(low_fire_n,:) = [];

  names = ["angle1" "angle2" "angle3" "angle4" "angle5" "angle6" "angle7" "angle8"];
  if ~isempty(test_data.decodedHandPos)
      reaching_angle = modelParameters.reaching_angle;
      %size(test_data.decodedHandPos)
      max_x =  modelParameters.(names(reaching_angle)).max_x;
      max_y =  modelParameters.(names(reaching_angle)).max_y;
      hand_position_x_past = test_data.decodedHandPos(1, size(test_data.decodedHandPos, 2))/max_x;
      hand_position_y_past = test_data.decodedHandPos(2, size(test_data.decodedHandPos, 2))/max_y;
  else
      reaching_angle = lda_classifier(test_data, modelParameters);
      modelParameters.reaching_angle = reaching_angle;
      max_x =  modelParameters.(names(reaching_angle)).max_x;
      max_y =  modelParameters.(names(reaching_angle)).max_y;
      hand_position_x_past = test_data.startHandPos(1)/max_x;
      hand_position_y_past = test_data.startHandPos(2)/max_y;
  end
  
  pca = modelParameters.(names(reaching_angle)).pca;
  lre_vel = modelParameters.(names(reaching_angle)).lre.vel;
  lre_pos = modelParameters.(names(reaching_angle)).lre.pos;
    
  end_spikes = size(test_data.spikes, 2);
  spikes_300ms = test_data.spikes(:, end_spikes-299:end_spikes);
  spikes_300ms = (spikes_300ms' - pca.mean)';
  spikes_300ms = reshape(pca.loadings'*spikes_300ms,[],1);
  spikes_300ms = [spikes_300ms; hand_position_x_past; hand_position_y_past];

  % Adding a column of ones to X to account for the intercept term
  X = [1; spikes_300ms]';

  if size(test_data.spikes, 2) == 320
      x = test_data.startHandPos(1);
      y = test_data.startHandPos(2);
  elseif size(test_data.spikes, 2) > 320 && size(test_data.spikes, 2) < 600
      pos = X * lre_pos;
      x = pos(1);
      y = pos(2);
  else
      vel = X * lre_vel;
      x = test_data.decodedHandPos(1,size(test_data.decodedHandPos,2))+vel(1)*0.02;
      y = test_data.decodedHandPos(2,size(test_data.decodedHandPos,2))+vel(2)*0.02;
      
  end
   
end

function [reaching_angle] = lda_classifier(data, modelParameters)
    
    rate = mean(data.spikes(:, 1:320), 2)';
    scores = rate * modelParameters.lda_coef' + modelParameters.lda_intercept;
    [~, reaching_angle] = max(scores, [], 2);
    
end