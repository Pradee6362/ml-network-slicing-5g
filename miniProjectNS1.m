% Load the data from Excel files
file1 = 'DeepSlice_ML_Model_Data_Public.xlsx';
file2 = 'DatasetNew1.xlsx';

% Read data from Excel sheets with the original column names preserved
data1 = readtable(file1, 'Sheet', 'ML_Inputs for Slicing', 'VariableNamingRule', 'preserve');
data2 = readtable(file2, 'VariableNamingRule', 'preserve');

% Preprocess the data
X = table2array(data2(:, 1:end-1)); % Features
y = table2array(data2(:, end));     

% Define the mapping of numeric labels to slice types
sliceMapping = [1, 2, 3]; % eMBB, URLLC, mIoT
sliceLabels = {'eMBB', 'URLLC', 'mIoT'};

% Convert numeric labels to categorical labels
y = categorical(y, sliceMapping, sliceLabels);

% Remove any rows with NaN or Inf values
validRows = ~any(isnan(X), 2) & ~any(isinf(X), 2);
X = X(validRows, :);
y = y(validRows);

% Split into training and testing sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.3);
XTrain = X(training(cv), :);
yTrain = y(training(cv), :);
XTest = X(test(cv), :);
yTest = y(test(cv), :);

% Define performance metric calculation
calculateMetrics = @(confMat) struct( ...
    'Precision', diag(confMat) ./ max(sum(confMat, 2), 1), ...
    'Recall', diag(confMat) ./ max(sum(confMat, 1)', 1), ...
    'F1Score', 2 * (diag(confMat) ./ max(sum(confMat, 2), 1)) .* ...
               (diag(confMat) ./ max(sum(confMat, 1)', 1)) ./ ...
              (diag(confMat) ./ max(sum(confMat, 2), 1) + diag(confMat) ./ max(sum(confMat, 1)', 1)), ...
    'Accuracy', sum(diag(confMat)) / sum(confMat(:)));

% Initialize models
models = {'KNN', 'CT', 'SVM', 'RF', 'GBM'};
predictions = cell(1, 5);
computationTime = zeros(1, numel(models));
numClasses = numel(sliceLabels);

% Store average metrics for graphing
modelMetrics = struct('Precision', [], 'Recall', [], 'F1Score', [], 'Accuracy', []);

% Train and evaluate each model
for i = 1:numel(models)
    tic;
    switch models{i}
        case 'KNN'
            model = fitcknn(XTrain, yTrain);
        case 'CT'
            model = fitctree(XTrain, yTrain);
        case 'SVM'
            model = fitcecoc(XTrain, yTrain);
        case 'RF'
            model = TreeBagger(100, XTrain, yTrain, 'OOBPrediction', 'On', 'Method', 'classification');
        case 'GBM'
            model = fitcensemble(XTrain, yTrain, 'Method', 'AdaBoostM2', 'NumLearningCycles', 100, 'Learners', 'Tree');
    end

    % Predict
    if strcmp(models{i}, 'RF')
        pred = predict(model, XTest);
        predictions{i} = categorical(pred, sliceLabels);
    else
        predictions{i} = predict(model, XTest);
    end

    computationTime(i) = toc;

    % Confusion matrix
    confMat = confusionmat(yTest, predictions{i}, 'Order', categorical(sliceLabels));
    confMat = confMat(sliceMapping, sliceMapping); % Ensure correct order

    % Metrics
    sliceResults = calculateMetrics(confMat);
    modelMetrics(i).Precision = mean(sliceResults.Precision);
    modelMetrics(i).Recall = mean(sliceResults.Recall);
    modelMetrics(i).F1Score = mean(sliceResults.F1Score);
    modelMetrics(i).Accuracy = sliceResults.Accuracy;

    % Display slice metrics
    sliceMetrics = table(sliceLabels', sliceResults.Precision, sliceResults.Recall, ...
        sliceResults.F1Score, repmat(sliceResults.Accuracy, numClasses, 1), ...
        'VariableNames', {'SliceType', 'Precision', 'Recall', 'F1Score', 'Accuracy'});

    disp(['Metrics for ' models{i} ':']);
    disp(sliceMetrics);

    % Confusion matrix plot
    figure;
    imagesc(confMat);
    colormap('cool');
    colorbar;
    confMatPercentage = 100 * confMat ./ sum(confMat, 2);
    [row, col] = size(confMat);
    for j = 1:row
        for k = 1:col
            text(k, j, sprintf('%d\n%.1f%%', confMat(j, k), confMatPercentage(j, k)), ...
                'HorizontalAlignment', 'center', 'Color', 'black');
        end
    end
    set(gca, 'XTick', 1:numClasses, 'XTickLabel', sliceLabels, ...
             'YTick', 1:numClasses, 'YTickLabel', sliceLabels);
    xlabel('Predicted');
    ylabel('Actual');
    title([models{i} ' Confusion Matrix (Counts and Percentage)']);
    axis square;
end

% Plot metric comparison for each slice
metricTypes = {'Precision', 'Recall', 'F1Score', 'Accuracy'};
for sliceIdx = 1:numClasses
    sliceMetricValues = zeros(numel(models), numel(metricTypes));
    for m = 1:numel(metricTypes)
        for modelIdx = 1:numel(models)
            confMat = confusionmat(yTest, predictions{modelIdx}, 'Order', categorical(sliceLabels));
            confMat = confMat(sliceMapping, sliceMapping);
            sliceResults = calculateMetrics(confMat);
            if sliceIdx <= length(sliceResults.(metricTypes{m}))
                sliceMetricValues(modelIdx, m) = sliceResults.(metricTypes{m})(sliceIdx);
            else
                sliceMetricValues(modelIdx, m) = NaN;
            end
        end
    end
    figure;
    hold on;
    for m = 1:numel(metricTypes)
        plot(1:numel(models), sliceMetricValues(:, m), '-o', 'LineWidth', 2, 'DisplayName', metricTypes{m});
    end
    set(gca, 'XTick', 1:numel(models), 'XTickLabel', models);
    legend('show', 'Location', 'northeastoutside');
    xlabel('Models');
    ylabel('Metric Scores');
    title(['Performance Comparison for ' sliceLabels{sliceIdx}]);
    grid on;
    hold off;
end

% Spectral Efficiency
dataRates = [1000, 100, 1]; % Mbps
bandwidths = [100, 20, 5]; % MHz
spectralEfficiency = dataRates ./ bandwidths;
for i = 1:numel(sliceLabels)
    fprintf('Spectral Efficiency for %s: %.2f bps/Hz\n', sliceLabels{i}, spectralEfficiency(i));
end

% Plot computation time with line and unique markers
colors = lines(numel(models));
markers = {'o', 's', 'd', '^', 'v'};
figure;
hold on;
plot(1:numel(models), computationTime, '-', 'Color', [0 0.45 0.74], ...
    'LineWidth', 2, 'DisplayName', 'Computation Time');
for i = 1:numel(models)
    plot(i, computationTime(i), ...
        'Marker', markers{i}, ...
        'MarkerSize', 10, ...
        'MarkerFaceColor', colors(i, :), ...
        'MarkerEdgeColor', 'k', ...
        'LineStyle', 'none', ...
        'DisplayName', models{i});
end
set(gca, 'XTick', 1:numel(models), 'XTickLabel', models);
xlabel('Model');
ylabel('Computation Time (seconds)');
title('Computation Time for Each ML Model');
legend('show', 'Location', 'northwest');
grid on;
hold off;

% Create computation time table
computationTimeTable = table(models', computationTime', ...
    'VariableNames', {'Model', 'ComputationTime_seconds'});
disp('Computation Time Table:');
disp(computationTimeTable);

% export to Excel
writetable(computationTimeTable, 'ComputationTimeResults.xlsx');
