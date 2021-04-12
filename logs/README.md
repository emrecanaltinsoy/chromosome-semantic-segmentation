### Scripts
  - computational_cost.py -> calculates the computational cost of each model
  - eval_others.py -> evaluates adaptive thresholding, histogram analysis, and the proposed cnn+bcn outputs
  - eval_threshold_vals.py -> finds the best thresholding values for the models
  - graphs.py -> outputs the tensorboard graphs into the graphs folder for each model
  - metrics.py -> calculates the evaluation metrics using the npy files of each model and writes the result in a yaml file
  - plot_losses.py -> plots the learning curves of every model
  - plot_threshold.py -> plots the threshold searching graphs for each model
  - print_best_threshold_metrics.py -> prints out the final comparison metrics
  - summary.py -> reads tensorboard scalar files and extracts the training and validation loss values for each model

### Yaml Files
  - dsc_scores.yaml -> contains test dsc scores for each model
  - evals.yaml -> contains TN, TP, FN, FP values for each model
  - metrics-m1.yaml -> contains chromosome prediction map evaluation metrics for each model
  - metrics-overall.yaml -> contains overall evaluation metrics for each model
 
### Graphs
graphs folder contains tensorboard graphs for each model

### Scalar
scalar folder contains:
  - tensorboard scalar files for each model. 
  - training arguments for each model
  - training and validation losses for each model
  - test evaluation numpy file for each model

### Threshold Evaluation
thresh_eval folder contains threshold value search results for each model

