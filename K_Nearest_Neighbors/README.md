# K nearest neighbors algorithm:

* Split the dataset into test and train.
* For every test point, calculate distance from all the training points.
* Sort the distances from each testing point in ascending order and pick the first 'k' elements.
* Calculate mode of the target variable for classification or average of the target variable for regression.


## Tuning:

* To tune the k nearest neighbors algorithm, pick different values of k (Square-root of the number of data points works in most of the cases)
* Plot error curve for each k value and choose the one with minimal error.
* Alternatively, use grid search or random search to select best metrics.
