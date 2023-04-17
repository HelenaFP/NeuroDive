%% Load data
clear
close all
load('monkeydata_training.mat')

%% 1) 
% Familiarize yourself with raster plots and for a single trial, compute and display a population
% raster plot. A population raster would have time (bins) on the x-axis and neural units on the yaxis

trial_nb = 1;
reaching_angle = 1;
nb_neurons = 1: size(trial(trial_nb,reaching_angle).spikes, 1);
max_time = size(trial(trial_nb,reaching_angle).spikes, 2);
time_axis = 1: max_time;

% Plot the raster plot
figure;
hold on;
for i = nb_neurons
    spike_train = trial(trial_nb,reaching_angle).spikes(i,:);
    for j = time_axis
        if spike_train(j)==1
            plot(time_axis(j) ,i,'k.');
        end
    end
end
xlabel('Time (ms)');
ylabel('Neuron number');
xline(300,'g')
axis tight
title('Raster Plot')
set(gcf,'color','w')

%% 2)
% Compute and display a raster plot for one neural unit over many trials (suggestion: try using
% different colours for different trials to help with the visualization).
neuron_nb = 1;
reaching_angle = 1;
nb_trials = 1: size(trial, 1);
% Plot the raster plot
figure;
hold on;
for i = nb_trials
    spike_train = trial(i,reaching_angle).spikes(neuron_nb,:);
    max_time = size(trial(i,reaching_angle).spikes, 2);
    time_axis = 1: max_time;
    for j = time_axis
        if spike_train(j)==1
            plot(time_axis(j) ,i,'k.');
        end
    end
end
xlabel('Time (ms)');
ylabel('Trial');
xline(300,'g')
axis tight
title('Raster Plot for single neuron')
set(gcf,'color','w')

%% 3)
% Familiarize yourself with what peri-stimulus time histograms (PSTHs) are and compute these
% for different neural units. 
neuron_nb = 1:98;
reaching_angle = 1:8;
nb_trials = 1: size(trial, 1);
% dimensions: reaching angle, neuron, time
psth = zeros(8, 98, 1000);

for tr = nb_trials
    for neuron = neuron_nb
        for angle = reaching_angle
            spike_train = trial(tr,angle).spikes(neuron,:);
            max_time = size(trial(tr,angle).spikes, 2);
            time_axis = 1: max_time;
            for j = time_axis
                if spike_train(j)==1
                    psth(angle, neuron,time_axis(j)) = psth(angle, neuron,time_axis(j)) + 1;
                end
            end
        end
    end
end

% Plot the PSTH for 10 neurons
fig = figure;
t_axis = 1:1000;
set(gcf, 'Position', get(0, 'Screensize'));
hold on

i = 1;
for angle = reaching_angle
    for neuron = 1:10
        subplot(8, 10, i)
        i = i+1;
        data = psth(angle, neuron, :);
        bar(t_axis,data(1,:));
        if mod(i-2, 10) ~= 0
            set(gca,'YTickLabel',[]);
        else
            ylabel(sprintf('Angle %d', angle));
        end
        if i < 12
            title(sprintf('Neuron %d', neuron));
        end
        if i < 70
            set(gca,'XTickLabel',[]);
        end
    end
end

% Give common xlabel, ylabel and title to your figure
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
titletext = "PSTH for 10 neurons and 8 reaching angles \n\n"
title(han, sprintf(titletext));

% axis tight
set(gcf,'color','w')


%% 4)
% Plot hand positions for different trials. Try to compare and make sense of cross-trial results.

figure
hold on;
nb_trials = 1: size(trial, 1);

colors = ["b", "g", "r", "k", "m", "c", "y", "#7E2F8E"];
for i = nb_trials
    for reaching_angle = 1:8
        plot(trial(i,reaching_angle).handPos(1,:), trial(i,reaching_angle).handPos(2,:),'Color', colors(reaching_angle))
    end
end
xlabel('x (cm)');
ylabel('y (cm)');
axis tight
title('Hand position')
set(gcf,'color','w')


%% 5)
% For several neurons, plot tuning curves for movement direction. Tuning curves measure
% directional preference of individual neural units.
% Get the tuning curve, by plotting the firing rate averaged across time and trials as a function of
% movement direction. To get an idea of the variability, compute the standard deviation across
% trials of the time-averaged firing rates.

% dimensions: reaching angle, neuron, trial number
firing_rate = zeros(8, 98, size(trial, 1));

neuron_nb = 1:98;
reaching_angle = 1:8;
nb_trials = 1: size(trial, 1);

for tr = nb_trials
    for neuron = neuron_nb
        for angle = reaching_angle
            firing_rate_trial = sum(trial(tr,angle).spikes(neuron,:))/size(trial(tr,angle).spikes(neuron,:),2);
            firing_rate(angle, neuron, tr)=firing_rate_trial;
        end
    end
end

mean_firing_rate = zeros(98, 8);
for neuron = neuron_nb
    for angle = reaching_angle
        mean_firing_rate(neuron, angle)=mean(firing_rate(angle, neuron, :));
    end
end

normalized_firing_rate = zeros(98, 8);
for neuron = neuron_nb
    f_max = max(mean_firing_rate(neuron, :));
    f_min = min(mean_firing_rate(neuron, :));
    denominator = f_max - f_min + eps;
    for angle = 1:8
        normalized_firing_rate(neuron, angle)= (mean_firing_rate(neuron, angle)-f_min) / denominator;
    end
end

% Plot the tuning curve for all neurons
fig = figure;
set(gcf, 'Position', get(0, 'Screensize'));
hold on
for neuron = neuron_nb
    subplot(10, 10, neuron)
    hold on
    for angle = reaching_angle
        plot(angle, normalized_firing_rate(neuron, angle),'-ob');
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
ylabel(han,'Normalized Firing rate');
xlabel(han,'Reaching Angle');
title(han,'Tuning curves for each neuron');

axis tight
set(gcf,'color','w')

%% 7)
% Implement the population vector algorithm and use it to predict arm
% movements.

% find prefered direction for each neuron
direction_angles = [30 70 110 150 190 230 310 350];
prefered_directions = zeros(1, 98);
for neuron = 1:98
    [maxValue, maxInd] = max(normalized_firing_rate(neuron,:), [], "all");
    prefered_directions(neuron) = direction_angles(maxInd);
end

figure
hold on
predicted_angles = zeros(100, 8);
colors = ["b", "g", "r", "k", "m", "c", "y", "#7E2F8E"];
for tr = 1:100
    for reaching_angle= 1:8
        v_pop = [0 ; 0];
        max_rate = max(mean_firing_rate(:));
        for neuron = 1:98
            rate = firing_rate(reaching_angle, neuron, tr);
            mean_neuron_rate = mean(mean_firing_rate(neuron, :));
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
for tr = 1:100
    if predicted_angles(tr, 8) < 10
        classified_angles(tr, 8) = 350;
    end
end

classification_accuracy = sum(classified_angles==direction_angles)/100;
mean_classification_accuracy = mean(classification_accuracy);
