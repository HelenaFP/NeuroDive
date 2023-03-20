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

    [coef, intercept] = calculate_lda(training_data);
    modelParameters.lda_coef = coef;
    modelParameters.lda_intercept = intercept;

    n_components = 13;
    names = ["angle1" "angle2" "angle3" "angle4" "angle5" "angle6" "angle7" "angle8"];
    for reaching_angle = 1:8
        modelParameters.(names(reaching_angle)) = struct;
        [loadings, mean_data, n_components] = calculate_pca(training_data, reaching_angle, n_components);
        modelParameters.(names(reaching_angle)).pca.loadings = loadings;
        modelParameters.(names(reaching_angle)).pca.mean = mean_data;
        modelParameters.(names(reaching_angle)).pca.n_components = n_components;

        [max_x, max_y] = get_max(training_data, reaching_angle);
        modelParameters.(names(reaching_angle)).max_x = max_x;
        modelParameters.(names(reaching_angle)).max_y = max_y;
        [beta_vel,beta_pos] = linear_regression(training_data, reaching_angle, loadings, mean_data, max_x, max_y);
        modelParameters.(names(reaching_angle)).lre.vel = beta_vel;
        modelParameters.(names(reaching_angle)).lre.pos = beta_pos;
        

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

function [coef, intercept] = calculate_lda(training_data)

    X = [];
    y = [];
    
    for tr = 1:size(training_data, 1)
        for direc = 1:8 
            X = [X mean(training_data(tr, direc).spikes(:, 1:320), 2)];
            y = [y direc];
        end
    end
    
    X = X';
    y = y';

    tol = 1e-4;
    [n_samples, n_features] = size(X);
    n_classes = 8;
    priors = ones(1, 8) / 8;
    
    class_labels = unique(y);
    means = zeros(length(class_labels), n_features);
    for i = 1:length(class_labels)
        c = class_labels(i);
        X_c = X(y == c, :);
        means(i, :) = mean(X_c);
    end
    
    Xc = zeros(size(X));
    from = 1;
    until = size(X, 1)/8;
    for i = 1:length(class_labels)
        c = class_labels(i);
        Xg = X(y == c, :);
        Xc(from:until, :) = bsxfun(@minus, Xg, means(i, :));
        from = until+1;
        until= until+size(X, 1)/8;
    end

    xbar = priors * means;
    std_dev = std(Xc);
    fac = 1 / (n_samples - n_classes);
    
    X = sqrt(fac)*(Xc./std_dev);
    
    [U, S, Vt] = svd(X, "vector");
    Vt= Vt';
    S = S';
    rank = sum(S > tol, 'all');
    scalings = (Vt(1:rank, :) ./ std_dev)' ./ S(1:rank);
    
    fac = 1 / (n_classes -1);
    X = ((sqrt(n_samples * priors * fac)) .* ((means - xbar)'))' * scalings;
    
    [~, S, Vt] = svd(X, "vector");
    S = S';

    rank = sum(S > tol * S(1), 'all');
    scalings = scalings * Vt(:, 1:rank);
    coef = (means - xbar) * scalings;
    intercept = -0.5 * sum(coef.^2, 2)' + log(priors);
    coef = coef * scalings';
    intercept = intercept - xbar * coef';
end


function [loadings, mean_data, n_components] = calculate_pca(train_data, reaching_angle, n_components)

    sizes = zeros(100, 1);
    for i = 1:size(train_data, 1)
        sizes(i, reaching_angle) = size(train_data(i,reaching_angle).spikes, 2);
    end
    
    max_size = max(sizes, [], 'all');
    
    % Calculate average spikes per neuron for that reaching angle
    n_neurons = size(train_data(1,1).spikes,1);
    average_spike_data = zeros(n_neurons, max_size);
    counts = zeros(n_neurons, max_size);
    for t = 1:size(train_data, 1)
        spikes = train_data(t, reaching_angle).spikes;
        average_spike_data(:, 1:size(spikes, 2)) = average_spike_data(:, 1:size(spikes, 2)) + spikes;
        counts(:, 1:size(spikes, 2)) = counts(:, 1:size(spikes, 2)) + 1;
    end
    
    average_spike_data = average_spike_data ./ counts;
    
    % subtract mean from data
    data = average_spike_data';
    mean_data = mean(data, 1);
    data = data - mean_data;
    
    % covariance matrix
    C = cov(data);
    
    % eigenvectors and eigenvalues
    [eigenvectors, eigenvalues] = eig(C);
    
    % select largest 32 eigenvalues and corresponding eigenvectors
    [eigenvalues, sorted_ids] = sort(diag(eigenvalues),'descend');
    eigenvectors = eigenvectors(:, sorted_ids);
    loadings = eigenvectors(:, 1:n_components);

end


function [beta_vel,beta_pos] = linear_regression(train_data, reaching_angle, loadings, mean_data, max_x, max_y)

    % Create training data
    spike_data_300 = [];
    position = [];
    velocity = [];
    vel= [];
    bin = 20;
    for trial_nb = 1:size(train_data,1)
        spikes = train_data(trial_nb, reaching_angle).spikes;
        for time = 1:20:size(train_data(trial_nb, reaching_angle).spikes,2)
            if size(spikes,2)>(320+time-1)
                spikes_300ms = spikes(:,time:300+time-1);
                spikes_300ms = (spikes_300ms' - mean_data)';
                spikes_300ms = reshape(loadings'*spikes_300ms,[],1);
                if time == 20
                    hand_position_x_past = train_data(trial_nb, reaching_angle).handPos(1,1)/max_x;
                    hand_position_y_past = train_data(trial_nb, reaching_angle).handPos(2,1)/max_y;
                else
                    hand_position_x_past = train_data(trial_nb, reaching_angle).handPos(1,300+time-20)/max_x;
                    hand_position_y_past = train_data(trial_nb, reaching_angle).handPos(2,300+time-20)/max_y;
                end
                spikes_300ms = [spikes_300ms; hand_position_x_past; hand_position_y_past];
                hand_position_x = train_data(trial_nb, reaching_angle).handPos(1,300+time);
                hand_position_y = train_data(trial_nb, reaching_angle).handPos(2,300+time);
                hand_position = [hand_position_x; hand_position_y];

                vel_x = (train_data(trial_nb, reaching_angle).handPos(1,time+bin+299)-train_data(trial_nb, reaching_angle).handPos(1,300+time))/0.02;
                vel_y = (train_data(trial_nb, reaching_angle).handPos(2,time+bin+299)-train_data(trial_nb, reaching_angle).handPos(2,300+time))/0.02;
                vel = [vel_x;vel_y];

                spike_data_300 = [spike_data_300, spikes_300ms];
                position = [position, hand_position];
                velocity = [velocity, vel];
            end
        end
    end

    X= spike_data_300';
    y1 = position';
    y2 = velocity';

    [m,n] = size(X);
    
    % Adding a column of ones to X to account for the intercept term
    X = [ones(m,1) X];
    
    % Calculate the Moore-Penrose pseudoinverse of X
    X_pinv = pinv(X);
    % Calculate the beta coefficients
    beta_pos = X_pinv * y1;
    beta_vel = X_pinv * y2;
   
end


function [max_x, max_y] = get_max(training_data, reaching_angle)

    max_x = 0;
    for tr = 1: size(training_data, 1)
        if abs(training_data(tr,reaching_angle).handPos(1,size(training_data(tr,reaching_angle).handPos, 2))) > max_x
            max_x = abs(training_data(tr,reaching_angle).handPos(1,size(training_data(tr,reaching_angle).handPos, 2)));
        end
    end
    
    max_y = 0;
    for tr = 1: size(training_data, 1)
        if abs(training_data(tr,reaching_angle).handPos(2,size(training_data(tr,reaching_angle).handPos, 2))) > max_y
            max_y = abs(training_data(tr,reaching_angle).handPos(2,size(training_data(tr,reaching_angle).handPos, 2)));
        end
    end
end