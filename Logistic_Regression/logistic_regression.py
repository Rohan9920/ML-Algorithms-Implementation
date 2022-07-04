import numpy as np
import matplotlib.pyplot as plt

class Logistic_Regression:


    def fit(self, X_train, y_train, learning_rate = 0.001, iterations = 500):
        """
            This function implements gradient descent algorithm. 
            It iteratively calculates theta using the derivative of the cost function.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.iterations = iterations
        m = X_train.shape[0]
        num_features = X_train.shape[1]
        self.theta = np.zeros(num_features) # Intializing theta for gradient descent. Would form an array of shape (m,)
        self.cost_list = []

        for i in range(iterations):
            self.cost_list.append(self.cost(X_train, y_train, self.theta, m)) # Store cost value for each record of theta for testing purpose.
            self.theta = self.theta - (learning_rate * self.derivative(X_train, y_train, self.theta, m))

    
    def predict(self, X_test):
        """
            This function would use the mimimized theta parameters and calculate the predicted value.
        """
        return np.round(self.calculate_sigmoid(np.dot(X_test, self.theta)))
    
    
    def calculate_sigmoid(self,x):
        """
            This function calculates the sigmoid value of 'x' array.
        """
        return 1/ (1 + np.exp(-x))

    
    def derivative(self, X_train, y_train, theta, m):
        """
            This function implements the derivative of the cost function.
        """
        return (1/m) * ((self.calculate_sigmoid(np.dot(X_train, theta)) - y_train).dot(X_train))


    def cost(self, X_train, y_train, theta, m):
        """
            This function implements the cost equation for logistic regression.
        """
        h = self.calculate_sigmoid(np.dot(X_train, theta))    
        return (-1/m) *((y_train).dot(np.log(h)) + (1-y_train).dot(np.log(1-h)))


    def visualize_cost(self):
        """
            This function is used to plot the cost function values against the number of iterations.
        """
        y_axis = [x for x in range(self.iterations)]
        plt.plot(y_axis,self.cost_list)
        plt.show()