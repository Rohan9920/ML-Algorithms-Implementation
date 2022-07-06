# Random Forest

Random Forest is an ensemble learning method for classification and regression that operates by constructing multiple decision trees.

## Algorithm steps:

- *Bootstrapping*: Choose random samples from the dataset with repetition.
- Build a decision tree using this *bootstrapped* dataset. You may not consider all the features while growing the decision tree.
- Repeat the above 2 steps 'n_estimators' times. n_estimators is the number of trees in the forest.
- While prediction, traverse each of the 'n_estimators' trees and output the majority class.

## Tuning:

Random Forest has various parameters that can be adjusted to get better results. Few of the important parameters are:

- n_estimators: Maximum no. of trees in the forest.
- max_features: Maximum no. of features to be considered at every split.
- max_depth: Maximum depth of each decision tree

## Files:

[random_forest.py](https://github.com/Rohan9920/ML-Algorithm-Implementations/blob/main/Random_Forests/random_forest.py): Reads the dataset and calls the functions.  
[rf_test.py](https://github.com/Rohan9920/ML-Algorithm-Implementations/blob/main/Random_Forests/rf_test.py): Includes all function definitions.

