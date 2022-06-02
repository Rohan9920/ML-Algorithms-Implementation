# importing libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Reading the dataset:

df = pd.read_csv('DataScienceProjects/ML_Implementations/K_Nearest_Neighbors/diabetes.csv')

# Checking null values:

print(df.isna().sum()) # No null values

# Separating dependent and independent variables. 
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Setting value of k
k=3                #Value of k is usually chosen as the square root of the number of elements. 

# Setting result array
y_pred = np.array([])

# Spltting the dataset in training and testing:

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=42)

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum(np.square(x1-x2)))

for i in range(X_test.shape[0]):
    distances = []
    for j in range(X_train.shape[0]):
        dist = euclidean_distance(np.array(X_test.iloc[i,:]), np.array(X_train.iloc[j,:]))
        distances.append(dist)
    distances = np.array(distances)
    dist_sort = np.argsort(distances)[:k]  #argsort function would return the indices of the first k elements sorted in ascending order
    
    y_pred = np.append(y_pred, max(y_train.iloc[[k for k in dist_sort]]))

results = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
print(results)



    



