import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree, Node

df = datasets.load_breast_cancer()
X, y = df.data , df.target
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)


dt = DecisionTree(max_depth=3)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# (np.sum(y_pred == y_test))/len(y_test)


