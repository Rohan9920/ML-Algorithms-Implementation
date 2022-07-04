import numpy as np
import pandas as pd
from logistic_regression import Logistic_Regression

df = pd.read_csv('Logistic_Regression/diabetes.csv')

# Separating dependent and independent variables. 
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Adding a column of 1's to the X dataframe
X = np.concatenate((np.ones(((np.shape(X)[0]),1)),X), axis = 1)

for i in range(0,df.shape[1]-1):
    df[df.columns[i]]= (df[df.columns[i]]-df[df.columns[i]].mean())/np.std(df[df.columns[i]])



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=42)

regressor = Logistic_Regression()
regressor.fit(X_train, y_train, iterations=5000, learning_rate=0.001)
y_pred = regressor.predict(X_test)



