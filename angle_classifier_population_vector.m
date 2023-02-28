%% Load data
clear all
close all
load('monkeydata_training.mat')

%% Calculate normalized tuning curves for all neurons with the first 320ms

% Firing rate only first 320 ms
% dimensions: reaching angle, neuron, trial number
firing_rate_320 = zeros(8, 98, size(trial, 1));

neuron_nb = 1:98;
reaching_angle = 1:8;
nb_trials = 1: size(trial, 1);

for tr = nb_trials
    for neuron = neuron_nb
        for angle = reaching_angle
            firing_rate_trial = sum(trial(tr,angle).spikes(neuron,1:320))/320;
            firing_rate_320(angle, neuron, tr)=firing_rate_trial;
        end
    end
end

mean_firing_rate_320 = zeros(98, 8);
for neuron = neuron_nb
    for angle = reaching_angle
        mean_firing_rate_320(neuron, angle)=mean(firing_rate_320(angle, neuron, :));
    end
end

% dimensions: neuron, reaching angle
normalized_firing_rate_320 = zeros(98, 8);
for neuron = neuron_nb
    f_max = max(mean_firing_rate_320(neuron, :));
    f_min = min(mean_firing_rate_320(neuron, :));
    denominator = f_max - f_min + eps;
    for angle = 1:8
        normalized_firing_rate_320(neuron, angle)= (mean_firing_rate_320(neuron, angle)-f_min) / denominator;
    end
end

%% Plot tuning curves
fig = figure
clear axis
set(gcf, 'Position', get(0, 'Screensize'));
hold on
for neuron = neuron_nb
    subplot(10, 10, neuron)
    hold on
    for angle = reaching_angle
        plot(angle, normalized_firing_rate_320(neuron, angle),'-ob');
    end
    if mod((neuron-1), 10)~=0
        set(gca,'YTickLabel',[]);
    end
    if neuron < 90
        set(gca,'XTickLabel',[]);
    end
end

% Give common xlabel, ylabel and title to your figure
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Normalized Firing rate - 320 ms');
xlabel(han,'Reaching Angle');
title(han,'Tuning curves for each neuron');

axis tight
set(gcf,'color','w')


%% Calculate population vector and plot it

% find prefered direction for each neuron
direction_angles = [30 70 110 150 190 230 310 350];
prefered_directions = zeros(1, 98);
for neuron = 1:98
    [maxValue, maxInd] = max(normalized_firing_rate_320(neuron,:), [], "all");
    prefered_directions(neuron) = direction_angles(maxInd);
end

figure
hold on
predicted_angles = zeros(100, 8);
colors = ["b", "g", "r", "k", "m", "c", "y", "#7E2F8E"];
% angles_legend = ["30", "70", "110", "150", "190", "230", "310", "350"];
for tr = 1:100
    for reaching_angle= 1:8
        v_pop = [0 ; 0];
        max_rate = max(mean_firing_rate_320(:));
        for neuron = 1:98
            rate = firing_rate_320(reaching_angle, neuron, tr);
            mean_neuron_rate = mean(mean_firing_rate_320(neuron, :));
            direction = exp(1i*(prefered_directions(neuron)/180 * pi));
            prefered_direction = [real(direction) ; imag(direction)];
            v_pop = v_pop + ((rate - mean_neuron_rate)/max_rate)* prefered_direction;
        end
        plot([0 v_pop(1)], [0 v_pop(2)], 'Color', colors(reaching_angle));
        predicted_angle = acos(dot(v_pop,[1 ;0])/norm(v_pop))*180/pi;
        if v_pop(2)<0
            predicted_angle = 360 -predicted_angle;
        end
        predicted_angles(tr, reaching_angle)=predicted_angle;
    end
end


% find angle class closest to predicted angle
classified_angles = zeros(100, 8);
for angl = 1:8
    [c index] = min(abs(predicted_angles(:, angl)-direction_angles), [], 2);
    classified_angles(:, angl) = direction_angles(index);
end
for tr = 100
    if predicted_angles(tr, 8) < 10
        classified_angles(tr, 8) = 350;
    end
end

classification_accuracy = sum(classified_angles==direction_angles)/100;
mean_classification_accuracy = mean(classification_accuracy);


