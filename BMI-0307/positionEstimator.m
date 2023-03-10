%%group name:NeuroDive
%%Team Members:Calista Yapeter, Haoxin Wu, Helena Ferreira Pinto, Varun Narasimhan


function [x, y, modelParameters,pre_dir] = positionEstimator(test_data, modelParameters)

  names = ["angle1" "angle2" "angle3" "angle4" "angle5" "angle6" "angle7" "angle8"];
  if ~isempty(test_data.decodedHandPos)
      reaching_angle = modelParameters.reaching_angle;
  else
      reaching_angle = lda_classifier(test_data, modelParameters);
      modelParameters.reaching_angle = reaching_angle;
  end
  pre_dir=reaching_angle;
  pca = modelParameters.(names(reaching_angle)).pca;
  lre_vel = modelParameters.(names(reaching_angle)).lre.vel;
  lre_pos = modelParameters.(names(reaching_angle)).lre.pos;
    
  end_spikes = size(test_data.spikes, 2);
  spikes_300ms = test_data.spikes(:, end_spikes-299:end_spikes);
  spikes_300ms = (spikes_300ms' - pca.mean)';
  spikes_300ms = reshape(pca.loadings'*spikes_300ms,[],1);
  X = [1; spikes_300ms]';

  % Adding a column of ones to X to account for the intercept term
  if size(test_data.spikes, 2) == 320
      x = test_data.startHandPos(1);
      y = test_data.startHandPos(2);
  elseif size(test_data.spikes, 2) > 320 && size(test_data.spikes, 2) < 600
      pos = X * lre_pos;
      x = pos(1);
      y = pos(2);
  else
      vel = X * lre_vel;
      if abs(vel(1))>100 && abs(vel(2))>100
        x = test_data.decodedHandPos(1,size(test_data.decodedHandPos,2))+vel(1)*0.02;
        y = test_data.decodedHandPos(2,size(test_data.decodedHandPos,2))+vel(2)*0.02;
      else
          x = test_data.decodedHandPos(1,size(test_data.decodedHandPos,2));
          y = test_data.decodedHandPos(2,size(test_data.decodedHandPos,2));
      end
  end
  
end

function [reaching_angle] = lda_classifier(data, modelParameters)
    
    rate = mean(data.spikes(:, 1:320), 2)';
    scores = rate * modelParameters.lda_coef' + modelParameters.lda_intercept;
    [~, reaching_angle] = max(scores, [], 2);
    
end



