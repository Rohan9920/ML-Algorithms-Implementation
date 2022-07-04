# Importing libraries
import numpy as np
import pandas as pd
from Multiple_Linear_Regression import Linear_regression

# Reading dataset
data = pd.read_csv('Linear_Regression/insurance.csv')
print(data.head())

# Converting categorical variables to numeric:

from sklearn.preprocessing import LabelEncoder
#sex
le = LabelEncoder()
le.fit(data.sex.drop_duplicates()) 
data.sex = le.transform(data.sex)
# smoker or not
le.fit(data.smoker.drop_duplicates()) 
data.smoker = le.transform(data.smoker)
#region
le.fit(data.region.drop_duplicates()) 
data.region = le.transform(data.region)

# Scaling the predictor variables.
# Required step for gradient descent. If variables are not scaled, the computation becomes too lengthy and 
# returns huge values eventually leading to overflow error.

for i in range(0,data.shape[1]-1):
    data[data.columns[i]]= (data[data.columns[i]]-data[data.columns[i]].mean())/np.std(data[data.columns[i]])

    
# Separating dependent and independent variables
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# Adding a column of 1's to the X dataframe
X = np.concatenate((np.ones(((np.shape(X)[0]),1)),X), axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=0.7, random_state=42)


model  = Linear_regression()
model.fit(X_train, y_train, iterations = 10, learning_rate = 0.01)
y_pred = model.predict(X_test)








