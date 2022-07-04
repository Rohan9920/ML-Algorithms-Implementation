# Linear Regression

This is a supervised algorithm predicting a continuous value using the best fit line.

## Algorithm steps: (Gradient Descent)

- Add a column of one to the dataset (for theta0).
- Scale the predictor variables (*Important step for gradient descent. If not scaled, the cost function keeps on increasing and eventually runs into overflow error*)
- Run a loop for n iterations.
- Minimize theta(regression co-efficients): In each iteration, calculate theta by using the derivative of the cost function.
- Use the minimzed theta parameter to predict the continuous variable.

## Tuning:

- Learning rate : Too less and algorithm converges slowly. Too high and algorithm may never find global minima.
- Iterations: Start with an arbitrary number. Verify if cost function remains stable as the iterations near to end.

