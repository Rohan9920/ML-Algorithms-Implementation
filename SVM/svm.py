import numpy as np
import matplotlib.pyplot as plt
class SVM:
    def __init__(self, learning_rate=0.01, lambda_param = 0.01, n_iters=1):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = 0
        self.b = 0
    

    def _init_weights_bias(self, X):
        n_features= X.shape[1]
        self.w = np.zeros(n_features)
        self.b=0
    
    
    def _get_cls_map(self,y):
        return np.where(y<=0, -1, 1)

    
    
    def _satisfy_constraint(self, x, idx):
        _linear_model = np.dot(x, self.w) + self.b
        return self.cls_map[idx]* _linear_model >=1

    
    def get_gradient(self, constraint, x, idx):
        # if data point lies on the correct side:
        if constraint:
            dw = self.lambda_param*self.w
            db=0
            return dw,db
        else:
            dw = self.lambda_param*self.w - np.dot(self.cls_map[idx],x)
            db = -self.cls_map[idx]
            return dw, db
    

    def _update_weights_bias(self, dw, db):
        self.w-=self.lr * dw
        self.b-=self.lr* db


    def fit(self,X,y):
        self._init_weights_bias(X)
        self.cls_map = self._get_cls_map(y)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                constraint = self._satisfy_constraint(x, idx)
                dw,db = self.get_gradient(constraint, x, idx)
                self._update_weights_bias(dw,db)

    
    def predict(self, X):
        estimate = np.dot( X, self.w) + self.b
        prediction = np.sign(estimate)
        return np.where(prediction==1,1,0)




