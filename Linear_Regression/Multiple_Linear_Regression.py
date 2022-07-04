import numpy as np
import matplotlib.pyplot as plt


class Linear_regression:
    

    def fit(self, X_train, y_train, iterations=500, learning_rate=0.001):
        """
            This function implements gradient descent algorithm. 
            It iteratively calculates theta using the derivative of the cost function.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.cost_list = []
        self.iterations = iterations
        self.len_rows = X_train.shape[0]
        self.len_features = X_train.shape[1]
        self.theta = np.zeros(self.len_features) # Intializing theta for gradient descent. Would form an array of shape (m,)
        
        for i in range(iterations):  
            self.cost_list.append(self.cost()) # Store cost value for each record of theta for testing purpose.
            self.theta = self.theta - learning_rate * self.derivative(self.theta, X_train, y_train, self.len_rows)
            

    
    def cost(self):
        """
            This function implements the cost equation of linear regression.
        """
        return 1/(2*(self.len_rows))* (np.sum(np.square(np.dot(self.X_train, self.theta) - self.y_train)))
        

    def derivative(self, theta, X_train, y_train, m):
        """
            This function implements the derivative of the cost function.
        """
        return (1/m)* ((np.dot(X_train, theta) - y_train).dot(X_train))
    
    
    def predict(self, X_test):
        """
            This function would use the mimimized theta parameters and calculate the predicted value.
        """
        return np.dot(X_test, self.theta)
    

    def visualize_cost(self):
        """
            This function is used to plot the cost function values against the number of iterations.
        """
        y_axis = [x for x in range(self.iterations)]
        print(y_axis)
        print(self.cost_list)
        plt.plot(y_axis,self.cost_list)
        plt.show()
        