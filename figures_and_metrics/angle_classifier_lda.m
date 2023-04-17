%% Load data
clear all
close all
load('monkeydata_training.mat')

%%
% Set random number generator
rng(2013);
ix = randperm(length(trial));

% Select training and testing data (you can choose to split your data in a different way if you wish)
training_data = trial(ix(1:70),:);
% testData = trial(ix(51:end),:);
test_data = trial(ix(71:end),:);

%%

training_data = trial(1:70,:);
% testData = trial(ix(51:end),:);
test_data = trial(71:end,:);
%%
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


%%

tol = 1e-4;
[n_samples, n_features] = size(X);
n_classes = 8;
priors = ones(1, 8) / 8;
max_components = min(n_classes - 1, size(X, 2));

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


explained_variance_ratio_ = (S.^2 / sum(S.^2));
explained_variance_ratio_ = explained_variance_ratio_(1:max_components);

rank = sum(S > tol * S(1), 'all');
scalings = scalings * Vt(:, 1:rank);
coef = (means - xbar) * scalings;
intercept_ = -0.5 * sum(coef.^2, 2)' + log(priors);
coef = coef * scalings';
intercept_ = intercept_ - xbar * coef';


%%
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

% Input:
% X: array-like of shape (n_samples, n_features)

% Output:
% X_new: ndarray of shape (n_samples, n_components) or
% (n_samples, min(rank, n_components))

[n_samples, n_features] = size(X);

X_new = (X - xbar) * scalings;

X_new = X_new(:, 1:max_components);

%%
figure;

colors = 'rgbcmykw';

for idx = 1:length(colors)
    c = colors(idx);
    target_name = idx;
    
    if c == 'w'
        color = "#7E2F8E";
    else
        color = c;
    end
    plot(X_new(y==idx, 1), X_new(y==idx, 2), 'o', 'Color', color, 'DisplayName', num2str(target_name));
    hold on;
end

legend('show');
title('LDA');

%%

X = [];
y = [];

for tr = 1:size(test_data, 1)
    for direc = 1:8 
        X = [X mean(test_data(tr, direc).spikes(:, 1:320), 2)];
        y = [y direc];
    end
end

X = X';
y = y';

% Input:
% X: {array-like, sparse matrix} of shape (n_samples, n_features)
% The data matrix for which we want to get the predictions.

% Output:
% y_pred: ndarray of shape (n_samples,)
% Vector containing the class labels for each sample.

scores = X * coef' + intercept_;

[~, indices] = max(scores, [], 2);

y_pred = class_labels(indices);


%%

classification_accuracy = sum(y_pred==y)/size(y,1) * 100;
