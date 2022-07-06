from dataclasses import replace
import numpy as np
from collections import Counter

def entropy(y):
    """
        This function is used to calculate entropy of each node.
        bincount will return occurences of each class label.
    """
    hist = np.bincount(y)
    ps = hist/ len(y)
    return -np.sum([p * np.log2(p) for p in ps if p>0])

class Node:
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, * , value=None):
        self.feature=feature
        self.threshold = threshold
        self.left= left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        """
            Helper class to determine if the node is a leaf node
        """
        return self.value is not None
    
class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        """
        Parameters:
        min_samples_split: Minimum no of samples to perform the split.
        max_depth: Maximum depth of the tree.
        n_feats: No of features to be considered at every level.
        """
        self.min_samples_split= min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        
        
    def fit(self, X_train, y_train):
        """
            This function is used to develop the tree using the _grow_tree function.
        """
        # If n_feats is not given, choose the number of features a column has and if given, verify if it is less than maximum no of features
        self.n_feats = X_train.shape[1] if not self.n_feats else min(self.n_feats, X_train.shape[1]) 
        self.root = self._grow_tree(X_train,y_train)


    def _grow_tree(self, X, y, depth=0):
        """
            This is a recursive function which grows the tree. It iterates through all the columns and column values
            and performs classification based on highest information gain.
        """
        
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):    
        # This is a stopping criteria. If any of the above conditions are satisfied, the node would be a leaf node. Leaf node 
        # will not have a left and right value.
            leaf_value = self._most_common_label(y) # Return the class with most no. of occurences.
            return Node(value=leaf_value)
            
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False) # Chooses columns randomly. If n_feats is not given, all columns will be chosen one by one. 

        # search
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs) # Column and column value that would return the highest information gain
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh) # Classify dataset based on best column and best column value
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right) # Returns best_feat and best_thresh for the first classification i.e the root node


    def  _best_criteria(self, X, y, feat_idxs):
        """
            This function determines the column and the column value that would give the highest information gain.
        """
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs: # Iterating through all columns
            X_column = X[: , feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold) # For every column value, calculate information gain.
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold
            
        return split_idx, split_threshold


    def _information_gain(self, y, X_column, split_thresh):
        """
            Calculates information gain for each column and threshold.
            Value between 0 and 1.
            0 means no new information can be gained from children. Parent is already a leaf node.
            1 means child nodes are 100% pure.
        """
            # parent Entropy
        parent_entropy = entropy(y)

            # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)
            
        if len(left_idxs) == 0 or len(right_idxs) ==0: # If elements in one of the nodes (either left or right) =0, no need to classify further. hence, information gain = 0.
            return 0
            
        # weighted avg child E
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        """
            This function is used to split the parent node into child nodes based on the column value currently in iteration.
            Note: For continuous variables, parent node elements will be classified on value being greater than or less than 
            the threshold value. 
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
        

    def predict(self, X):
        # traverse tree
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


