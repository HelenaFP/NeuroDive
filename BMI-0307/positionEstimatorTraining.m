%%group name:NeuroDive
%%Team Members:Calista Yapeter, Haoxin Wu, Helena Ferreira Pinto, Varun Narasimhan
function [modelParameters] = positionEstimatorTraining(training_data)
    modelParameters = struct;

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

        [beta_vel,beta_pos] = linear_regression(training_data, reaching_angle, loadings, mean_data);
        modelParameters.(names(reaching_angle)).lre.vel = beta_vel;
        modelParameters.(names(reaching_angle)).lre.pos = beta_pos;
    end
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
    average_spike_data = zeros(98, max_size);
    counts = zeros(98, max_size);
    for t = 1:size(train_data, 1)
        spikes = train_data(t, reaching_angle).spikes;
        average_spike_data(:, 1:size(spikes, 2)) = average_spike_data(:, 1:size(spikes, 2)) + spikes;
        counts(:, 1:size(spikes, 2)) = counts(:, 1:size(spikes, 2)) + 1;
    end
    
    average_spike_data = average_spike_data ./ counts;
    
    % subtract mean from data
    data = average_spike_data';
    mean_data = mean(data, 1);
%     sd = std(data,1);
%     data = (data - mean_data)./sd;
    
    % covariance matrix
    C = cov(data);
    
    % eigenvectors and eigenvalues
    [eigenvectors, eigenvalues] = eig(C);
    
    % select largest 32 eigenvalues and corresponding eigenvectors
    [eigenvalues, sorted_ids] = sort(diag(eigenvalues),'descend');
%     plot(eigenvalues);
    eigenvectors = eigenvectors(:, sorted_ids);
    loadings = eigenvectors(:, 1:13);

end


function [beta_vel,beta_pos] = linear_regression(train_data, reaching_angle, loadings, mean_data)

    % Create training data
    spike_data_300 = [];
    velocity = [0;0];
    position = [];
    vel=[];
    bin = 20;
    for trial_nb = 1:size(train_data,1)
        spikes = train_data(trial_nb, reaching_angle).spikes;
        time_len = 1:bin:size(train_data(trial_nb, reaching_angle).spikes,2);
        for time = time_len
            if size(spikes,2)>(320+time-1)
                spikes_300ms = spikes(:,time:time+299);
                spikes_300ms = (spikes_300ms' - mean_data)';
                spikes_300ms = reshape(loadings'*spikes_300ms,[],1);
                if time~=1
                    vel_x = (train_data(trial_nb, reaching_angle).handPos(1,time+bin+299)-train_data(trial_nb, reaching_angle).handPos(1,time+300))/0.02;
                    vel_y = (train_data(trial_nb, reaching_angle).handPos(2,time+bin+299)-train_data(trial_nb, reaching_angle).handPos(2,time+300))/0.02;
                    vel = [vel_x;vel_y];
                end
                hand_position_x = train_data(trial_nb, reaching_angle).handPos(1,300+time-1);
                hand_position_y = train_data(trial_nb, reaching_angle).handPos(2,300+time-1);
                
                hand_position = [hand_position_x; hand_position_y];      

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
