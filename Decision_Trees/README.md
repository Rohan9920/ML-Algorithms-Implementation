# Decision Tree

Decision Tree is a supervised learning algorithm that uses a tree-like model of decisions to predict continuous or discrete variable.

## Algorithm steps:

- Feature values need to be categorical. If the values are continuous then they are discretized prior to building the model.
- Iterate through each feature and each distinct value for each feature and calculate information gain for the node.
```
Information Gain = Entropy (Parent) - Entropy (Child) {Gini impurity could also be used to calculate Information gain}
```
- Choose the feature and value with the highest information gain.
- Classify the dataset based on the above chosen criteria.
- Continue this process until any of the following conditions is satisfied:
  - Maximum depth is reached. 
  - Only 1 distinct label is left to classify.
  - No. of samples is less than the minimum no of samples specified.
- For prediction, traverse through the entire tree based on the input and return the class of the final leaf encountered.

## Tuning:

There are various parameters available for tuning decision trees. Some of them are:

- max_depth: Determines the maximum depth a decision tree can achieve. Higher value may lead to overfitting.
- min_samples_split: Minimum samples required to split a node.
- max_features: Maximum features to be considered while deciding the split.

## Files:

[dr_test.py](https://github.com/Rohan9920/ML-Algorithm-Implementations/blob/main/Decision_Trees/dr_test.py): Reads the dataset and calls the functions.  
[decision_tree.py](https://github.com/Rohan9920/ML-Algorithm-Implementations/blob/main/Decision_Trees/decision_tree.py): Includes all function definitions.
