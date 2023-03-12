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

    n_components = 22;
    names = ["angle1" "angle2" "angle3" "angle4" "angle5" "angle6" "angle7" "angle8"];
    for reaching_angle = 1:8
        modelParameters.(names(reaching_angle)) = struct;
        [loadings, mean_data, n_components] = calculate_pca(training_data, reaching_angle, n_components);
        modelParameters.(names(reaching_angle)).pca.loadings = loadings;
        modelParameters.(names(reaching_angle)).pca.mean = mean_data;
        modelParameters.(names(reaching_angle)).pca.n_components = n_components;

        [beta] = linear_regression(training_data, reaching_angle, loadings, mean_data);
        modelParameters.(names(reaching_angle)).lre = beta;
    end
end

function low_fire_n = get_low_firing_neurons(training_data)
    firing_rate = [];
    firing_rate_angle = [];
    for n = 1:length(training_data)
        for k = 1:8
            firing_rate = [firing_rate mean(training_data(n,k).spikes(:,:),2)];
        end
        firing_rate_angle =[firing_rate_angle mean(firing_rate,2)];
    end
    average_firing_rate = mean(firing_rate_angle,2); 

    [low_rates,low_fire_n] = mink(average_firing_rate,15);
end

function [coef, intercept] = calculate_lda(training_data)

    X = [];
    y = [];
    
    for tr = 1:size(training_data, 1)   % loop over no. of trials
        for direc = 1:8     % loop over direction
            % X stores the mean firing rate over time for the first 320s
            % of all neurons for one trial
            X = [X mean(training_data(tr, direc).spikes(:, 1:320), 2)]; 
            y = [y direc];
        end
    end
    % X is a n_neuron x (n_direction x n_trial) array, containing 
    % the mean firing rate of each neuron for each direction in each trial

    % transpose the arrays
    % Dimensions of X = (n_direction x n_trial) x n_neuron
    X = X';
    y = y';

    tol = 1e-4;
    [n_samples, n_features] = size(X);
    n_classes = 8;
    % priors: same probability of being in each class (1/n_classes)
    priors = ones(1, 8) / 8;
    
    class_labels = unique(y); % ensures no repeated class labels
    means = zeros(length(class_labels), n_features);
    for i = 1:length(class_labels)
        c = class_labels(i);
        % find the indexes where the direction y is the same as the current
        % class label. X_c should be n_trials x n_neuron array
        X_c = X(y == c, :);
        
        % Find the mean firing rate over trials for a particular direction,
        % for each neuron. mean(X_c) is a 1xn_neuron array containing
        % this information. Stack these together to get means, which is a 
        % n_classes x n_neurons array containing the mean firing rate over 
        % time and trials for all neurons in each direction
        means(i, :) = mean(X_c);
    end
    
    Xc = zeros(size(X));
    from = 1;   % first trial of window
    until = size(X, 1)/8;   % last trial of window
    for i = 1:length(class_labels)
        c = class_labels(i);
        Xg = X(y == c, :);
        % https://www.mathworks.com/help/matlab/ref/bsxfun.html
        % subtract the mean firing rate over time and trials from the
        % original data (for one direction)
        Xc(from:until, :) = bsxfun(@minus, Xg, means(i, :));
        
        % shift to next set of trials and repeat 
        from = until+1;
        until= until+size(X, 1)/8;
    end
    
    % xbar is the mean accounting for class prior probabilities
    xbar = priors * means;
    std_dev = std(Xc);
    % n_samples is n_directions * n_trials
    fac = 1 / (n_samples - n_classes);
    
    X = sqrt(fac)*(Xc./std_dev);
    
    % U is the matrix of orthonormal eigenvectors of AA^T
    [U, S, Vt] = svd(X, "vector");
    % Vt' is a matrix containing the orthonormal eigenvectors of A^{T}A
    Vt= Vt';
    % S is the diagonal matrix of the singular values which are the square roots of the eigenvalues of A^{T}A                
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
    for i = 1:size(train_data, 1)   % loop over number of trials
        sizes(i, reaching_angle) = size(train_data(i,reaching_angle).spikes, 2); % get the number of time points in each trial
    end
    
    max_size = max(sizes, [], 'all');   % record the highest number of spikes/time points for a trial
    
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
    loadings = eigenvectors(:, 1:22);

end

% beta are the coefficients of the regression equation
function [beta] = linear_regression(train_data, reaching_angle, loadings, mean_data)

    % Create training data
    spike_data_300 = [];
    position = [];
    t_window = 250;
    for trial_nb = 1:size(train_data,1)
        spikes = train_data(trial_nb, reaching_angle).spikes;
        for time = 1:15:675
            if size(spikes,2)>(t_window+time-1)
                % get the 300ms of time data, shifting by 20ms each time
                spikes_300ms = spikes(:,time:t_window+time-1);
                spikes_300ms = (spikes_300ms' - mean_data)';
                spikes_300ms = reshape(loadings'*spikes_300ms,[],1);
               
                % get the final hand position after time interval
                hand_position_x = train_data(trial_nb, reaching_angle).handPos(1,t_window+time-1);
                hand_position_y = train_data(trial_nb, reaching_angle).handPos(2,t_window+time-1);
                hand_position = [hand_position_x; hand_position_y];
                
                % add window of spikes to spike_data_300
                spike_data_300 = [spike_data_300, spikes_300ms];
                position = [position, hand_position];
            end
        end
    end

    X= spike_data_300';
    y = position';

    [m,n] = size(X);
    
    % Adding a column of ones to X to account for the intercept term
    X = [ones(m,1) X];
    
    % Calculate the Moore-Penrose pseudoinverse of X
    X_pinv = pinv(X);
    % Calculate the beta coefficients
    beta = X_pinv * y;
   
end
