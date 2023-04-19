function [modelParameters] = knn_train(modelParameters,train_rates,n_timepoints)

modelParameters.knn = struct;
names = ["angle1" "angle2" "angle3" "angle4" "angle5" "angle6" "angle7" "angle8"];
[trials_per_angle, nangles] = size(train_rates);
neighbours = [];
angles = [];


for angle = 1:nangles
    for tr = 1:trials_per_angle

        t_length = size(train_rates(tr,angle).rates,2);

        % first n_timepoints points
        fire = train_rates(tr,angle).rates(:,1:n_timepoints);

        neighbours = [neighbours, mean(fire,2)];
        angles = [angles, angle];
    end

end

modelParameters.knn.neighbours = neighbours;
modelParameters.knn.angles = angles;



end