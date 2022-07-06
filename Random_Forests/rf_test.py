import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Random_Forests.random_forest import RandomForest

df = datasets.load_breast_cancer()
X, y = df.data , df.target
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)


rf = RandomForest(n_trees=3, max_depth=3)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)



## Working of counter function:

# counter = Counter(np.array([2,2,1,1]))
# most_common = counter.most_common() 

## Above line returns a tuple with most common element and no of elements. For the above example, [(2, 2), (1, 2)] is returned 
## Choosing first tuple and first element:

#most_common = counter.most_common(1)[0][0] 
#most_common







