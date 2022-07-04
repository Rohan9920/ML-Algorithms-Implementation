# Logistic Regression #

This is a supervised classification algorithm. It predicts the class of the dependent variable by fitting a sigmoid function.

## Algorithm steps (Gradient Descent)

- Add a column of one to the dataset (for theta0).
```
np.concatenate(np.ones((m,1)), X)

```
- Scale the predictor variables (Important step for gradient descent. If not scaled, the cost function keeps on increasing and eventually runs into overflow error)
- Run a loop for n iterations.
- In each iteration, calculate theta by using the derivative of the cost function.
- Use the minimized theta parameter to predict the class of dependent variable using sigmoid function.

## Tuning:

- Learning rate : Too less and algorithm converges slowly. Too high and algorithm may never find global minima.
- Iterations: Start with an arbitrary number. Verify if cost function remains stable as the iterations near to end.

## Files:

[logistic_test.py](https://github.com/Rohan9920/ML-Algorithm-Implementations/blob/main/Logistic_Regression/logistic_test.py): Reads the dataset and calls the functions.  
[logistic_regression.py](https://github.com/Rohan9920/ML-Algorithm-Implementations/blob/main/Logistic_Regression/logistic_regression.py): Includes all function definitions.
