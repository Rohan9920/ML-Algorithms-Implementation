# importing libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Reading the dataset:

df = pd.read_csv('DataScienceProjects/ML_Implementations/diabetes.csv')
print(df.head())

# Checking null values:

print(df.isna().sum()) # No null values

# Separating dependent and independent variables. 
X = df[:,:]
y = df[,]

# Spltting the dataset in training and testing:

X_train, X_test, y_train, y_test = train_test_split(df)