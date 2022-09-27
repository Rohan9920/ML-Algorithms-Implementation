# Support Vector Machines

SVM is a supervised algorithm that is used for both regression and classification. It finds a hyperplane such that the distance between 2 classes is maximized.

# Algorithm steps (for soft svm) 

- Eqn of a plane is given by: 

```math  
wx+b = 0
```
w and b needs to be calculated to return the optimal plane separating the 2 classes. Initialize w and b to 0.
- For each point check if it satisfies the constraint: 

```math
                        y(wx+b) \leq =1
```
If yes, calculate, <br></br>
```math  
    dw = &#955;w
```
```math
    db = 0
```    
If no, calculate <br></br>
```math
    dw = &#955;w - yx
```
```math
  db = -y
```
 - After every iteration, updated w and b value by:
 ```math
    w = w-lr*dw
 ```
 ```math
    b = b-lr*dw
 ```
 - Repeat the above steps for n iterations.
 - Use the calculated w and b to draw the plane that would best separate the 2 classes.
 
 
 *Note: If linear SVM does not work kernels are used. Kernels are special functions that allow us to reap the benefits of higher dimensional vector space without actually transforming the vectors to the higher dimension.*
 


