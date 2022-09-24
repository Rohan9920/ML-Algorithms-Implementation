# Support Vector Machines

SVM is a supervised algorithm that is used for both regression and classification. It finds a hyperplane such that the distance between 2 classes is maximized.

# Algorithm steps (for soft svm) 

- Eqn of a plane is given by wx+b = 0. w and b needs to be calculated to return the optimal plane separating the 2 classes. Initialize w and b to 0.
- For each point check if it satisfies the constraint: 

```math
                        y(wx+b) \leq =1
```
    If yes, calculate
    w = 
    b = 
    If no, calculate
    w = 
    b =
 - Repeat the above steps for n iterations.
 - Use the calculated w and b to draw the plane that would best separate the 2 classes.
 
 
 *Note: If linear SVM does not work kernels are used. Kernels are special functions which allow us to reap the benefits of higher dimensional vector space without actually transforming the vectors to the higher dimension.*
 
 
