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