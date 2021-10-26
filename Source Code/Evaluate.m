function EVAL = Evaluate(ACTUAL,PREDICTED)
% This fucntion evaluates the performance of a classification model by
% calculating the common performance measures: Accuracy, Sensitivity,
% Specificity, Precision, Recall, F-Measure, G-mean.
% Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
% Output: EVAL = Row matrix with all the performance measures
if size(unique(ACTUAL))== size(unique(PREDICTED))
    PREDICTED = (calculate_true_labels(PREDICTED',ACTUAL))';
    [c_matrix,Result,RefereceResult]= confusion.getMatrix(ACTUAL',PREDICTED');
    
    % Compute the unadjusted rand index
    ri_unadjusted = rand_index(ACTUAL', PREDICTED');
    
    % Compute the adjusted rand index
    ri_adjusted = rand_index(ACTUAL', PREDICTED', 'adjusted');
    EVAL = [Result.Accuracy,ri_adjusted, Result.Precision,Result.Sensitivity];
else
    EVAL = [inf,inf, inf,inf];
end


