function [modelParameters] = positionEstimatorTraining(training_data)

    [trials_per_angle, nangles] = size(training_data);
    modelParameters = struct;
    
%%%%%%%%%%%%%%%%%%%%%%%% PUT IN NEW BINS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    train_new_bins = struct;
    
    new_bin = 10; % ms
    
    for i = 1:nangles
        for j = 1:trials_per_angle

            all_spikes = training_data(j,i).spikes; % spikes is no neurons x no time points
            nneurons = size(all_spikes,1);
            t_length = size(all_spikes,2);
            t_new = 1: new_bin : t_length; % New time point array
            handPos = training_data(j,i).handPos(1:2,:);
            spikes = zeros(nneurons,numel(t_new)-1);

            for k = 1 : numel(t_new) - 1 % get rid of the paddded bin
                spikes(:,k) = sum(all_spikes(:,t_new(k):t_new(k+1)-1),2);
            end

            spikes = sqrt(spikes);
            binned_handPos = handPos(:,t_new);
            
            train_new_bins(j,i).spikes = spikes;
            train_new_bins(j,i).handPos = binned_handPos;
        end
    end
    
    
%%%%%%%%%%%%%%%%%%%%%%%%% FIRING RATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
% Using a gaussian kernel

[train_rates] = gaussian(train_new_bins,new_bin);


    
%%%%%%%%%%%%%%%%%%%%%%%%%%%% REMOVE LOW FIRING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    % find and remove low firing neurons from training data, and store
    % these neurons (indices) to be removed from test data
    min_fire = 5.0;
    [low_fire_n,modelParameters] = get_low_firing_neurons(modelParameters,train_rates,min_fire);
    modelParameters.low_fire_n = low_fire_n;
    for n = 1:trials_per_angle
        for k = 1:nangles
            train_rates(n,k).spikes(low_fire_n,:) = [];
            train_rates(n,k).rates(low_fire_n,:) = [];
        end
    end
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%% PCA - LDA FOR ANGLE CLASSIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    nLDA = 5;
    names = ["angle1" "angle2" "angle3" "angle4" "angle5" "angle6" "angle7" "angle8"];
    
    

    [W,opt_proj] = calculate_lda(train_rates,nLDA,new_bin);
    %[coef, intercept] = calculate_lda(train_rates,nPCA,nLDA);
    %modelParameters.lda_coef = coef;
    %modelParameters.lda_intercept = intercept;
    modelParameters.ldaW = W;
    modelParameters.lda_nlda = nLDA;
    modelParameters.lda_opt_proj = opt_proj;
    

%%%%%%%%%%%%%%%%%%%%%%%%%%% LINEAR REGRESSION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Gaussian kernel full set of spikes

[train_full] = gaussian(training_data,20.0);

    n_PCA_components = 25;
    for reaching_angle = 1:nangles
        modelParameters.(names(reaching_angle)) = struct;
        [loadings, mean_data] = calculate_pca(train_full, reaching_angle, n_PCA_components);
        modelParameters.(names(reaching_angle)).pca.loadings = loadings;
        modelParameters.(names(reaching_angle)).pca.mean = mean_data;
        modelParameters.(names(reaching_angle)).pca.n_components = n_PCA_components;

        [beta] = linear_regression(train_full, reaching_angle, loadings, mean_data,20.0);
        modelParameters.(names(reaching_angle)).lre = beta;
    end
end

function [lowFire,modelParameters] = get_low_firing_neurons(modelParameters,training_data,min_fire)  
    

    %low_fire_n = {};
    
    nNeurons = length(training_data(1,1).spikes);

    [ntrials,nangles] = size(training_data);

    angle_rates = zeros(nNeurons,ntrials*nangles);

    % calculate which neurons to remove based on up to 560 first!
      
    k = 1;
    for i = 1:ntrials    
        for j=1:nangles
            angle_rates(:,k) = mean(training_data(i,j).rates,2);
            k = k + 1;
        end
    end
    
    
    angle_rates_all = mean(angle_rates,2);

    lowFire = [];
    for x = 1: nNeurons
        if angle_rates_all(x) < min_fire
            lowFire = [lowFire,x];

        end
    end
    
    modelParameters.lowFire = lowFire; 
    
end

function [W,opt] = calculate_lda(training_data,nLDA,new_bin)
    
    ntrials = size(training_data, 1);
    trimmer = 320/new_bin - 1; % make the trajectories the same length
    X = zeros([size(training_data(1,1).rates,1)*trimmer,ntrials*8]);
    
    
    
    nNeurons = size(training_data(1,1).rates,1);
    
        % need to get (neurons x time)x trial
    for i = 1: 8
        for j = 1: ntrials
            for k = 1: trimmer
                X(nNeurons*(k-1)+1:nNeurons*k,ntrials*(i-1)+j) = training_data(j,i).rates(:,k);     
            end
        end
    end
    
    % Scatter matrices
    mi = zeros(size(X,1),8);
    
    for i = 1: 8
        mi(:,i) =  mean(X(:,ntrials*(i-1)+1:i*ntrials),2);
    end
    
    % Scatter between classes matrix
    sb = (mi - mean(X,2))*(mi - mean(X,2))';
    
    % Scatter within class matrix
    
    sw = zeros(size(sb));
    
    for i=1:8
        sw = sw + (X-mi(:,i))*(X-mi(:,i))';
    end       
    
    % PCA components
    
    thresh_var = 0.99;
    
    [PC, ~, nPCA] = get_pca(X, thresh_var);
    
   
    [eVeLDA, eVaLDA] = eig(((PC'*sw*PC)^-1 )*(PC'*sb*PC));
    [~,sort_inds] = sort(diag(eVaLDA),'descend');
    
    % Get most important projections of PCA onto LDA
    opt = PC*eVeLDA(:,sort_inds(1:nLDA));
    
    % Most Discriminant Feature
    W = opt'*(X - mean(X,2));
    
    colors = {[1 0 0],[0 1 1],[1 1 0],[0 0 0],[0 0.75 0.75],[1 0 1],[0 1 0],[1 0.50 0.25]};
    figure
    hold on
    for i=1:8
        plot(W(1,ntrials*(i-1)+1:i*ntrials),W(2,ntrials*(i-1)+1:i*ntrials),'o','Color',colors{i},'MarkerFaceColor',colors{i},'MarkerEdgeColor','k')
        hold on
    end

    legend('1','2','3','4','5','6','7','8');

end

function [PC, evals, nPCA] = get_pca(X, thresh_var)

    X_centred = (X - mean(X,2))./std(X,0,2);
    % calculate the covariance matrix
    covX = cov(X);
    % get eigenvalues and eigenvectors
    [evects, evals] = eig(covX);
    % Eigenpairs in descending order
    [evals,sorted] = sort(diag(evals),'descend');
    evects = evects(:,sorted);
    % Get PCs
    PC = X_centred*evects;
    
    % Extract the important PCs. 0.95 threshold
    
    total_eval = sum(evals);
    
    min_total = thresh_var*total_eval;
    
    curr_total = 0;
    
    nPCA = 0;
    while curr_total < min_total
        nPCA = nPCA + 1;
        curr_total = curr_total + evals(nPCA);
    end
        
    PC = PC(:,1:nPCA);
    
    % normalisation
    PC = PC./sqrt(sum(PC.^2));

end
    
function [loadings, mean_data] = calculate_pca(train_data, reaching_angle, n_components)

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
    loadings = eigenvectors(:, 1:n_components);

end

% beta are the coefficients of the regression equation
function [beta] = linear_regression(train_data, reaching_angle, loadings, mean_data,new_bin)
    % Create training data  
    
    spike_data_320 = [];
    position = [];
    for trial_nb = 1:size(train_data,1)
        spikes = train_data(trial_nb, reaching_angle).spikes;
        for time = 1:size(spikes,2)-1-320/new_bin
             % get the 320ms of time data, shifting by 20ms each time
             spikes_320ms = spikes(:,time:time-1+320/new_bin);
             spikes_320ms = (spikes_320ms' - mean_data)';
             spikes_320ms = loadings'*spikes_320ms;
               
             % get the final hand position after time interval
             hand_position_x = train_data(trial_nb, reaching_angle).handPos(1,time:time-1+320/new_bin);
             hand_position_y = train_data(trial_nb, reaching_angle).handPos(2,time:time-1+320/new_bin);
             hand_position = [hand_position_x; hand_position_y];
                
             % add window of spikes to spike_data_300
             spike_data_320 = [spike_data_320, spikes_320ms];
             position = [position, hand_position];
        end
    end

    X= spike_data_320';
    
    y = position';

    [m,~] = size(X);
    
    
    clear spike_data_320
    clear position
    clear hand_position_x
    clear hand_position_y
    clear spikes_320ms
    
    
    % Adding a column of ones to X to account for the intercept term
    X = [ones(m,1) X];
    
    % Calculate the Moore-Penrose pseudoinverse of X
    X_pinv = pinv(X);
    % Calculate the beta coefficients
    beta = X_pinv * y;
   
end


function [train_rates] = gaussian(train_new,new_bin)

train_rates = struct;
    
    scale_win = 50;
    
    std = scale_win/new_bin;
    w = 10*std;
    
    alpha = (w-1)/(2*std);
    temp1 = -(w-1)/2 : (w-1)/2;
    gausstemp = exp((-1/2) * (alpha * temp1/((w-1)/2)) .^ 2)';
    gaussian_window = gausstemp/sum(gausstemp);
    
    for i = 1: size(train_new,2)

        for j = 1:size(train_new,1)
            
            conv_rates = zeros(size(train_new(j,i).spikes,1),size(train_new(j,i).spikes,2));
            
            for k = 1: size(train_new(j,i).spikes,1)
                
                % COnvolve with gaussian window
                conv_rates(k,:) = conv(train_new(j,i).spikes(k,:),gaussian_window,'same')/(new_bin/1000);
            end
            
            train_rates(j,i).rates = conv_rates;
            train_rates(j,i).handPos = train_new(j,i).handPos;
            train_rates(j,i).spikes = train_new(j,i).spikes;
        end
    end  

end




