#example solution for ann-ml lectures
###simple regression analysis
- simple_linear_regression_example_ANN.py
  - [x] Very basic solution as brute forcing all possible models.
  	1. brute force all possible solutions.
  	2. compare RSS of solution models.
  	3. print top #(5) solutions.
  - _We can compare models with approved solution formula of field of statistics/economics seen on graph._
  ![program output](https://github.com/ibo52/ann-ml/blob/master/out%20imgs/result_low_precision.png)
  ***
- haloLearn.py
  - [x] a simple algorithm for prediction problem. Can be improved by comparing two functions on a step, thus reduce functions that exceed turning points.
    1. determine coefficient(mean slope) of dots ( SUM(y)/SUM(x) )
    2. determine a stepsize that how long it re-predict
      3. on every re-prediction, determine mean constant value by coefficient (b=y- ax)
      4. calculate RSS by estimated function values.
      5. if estimated RSS is better than RSS before; continue in this direction.
      6. if not; change direction to revert and decrease precision to get closer to better model.
  - We see how function works by plotting steps on figure. And compare with statistics formula.
  ![program output](https://github.com/ibo52/ann-ml/blob/master/out%20imgs/output2.gif)
  ***
- haloLearn-multi.py
  - [x] polinomial regression. Curving the line to make more accurate data model. Same algorithm with haloLearn. Have logarithmic velocity stepping precision.
  - _higher orders can be added by specifying power parameter of function._
    1. Calculate simple linear regression.
    2. Calculate the curve function(x^2) coefficient(w2) to make model much more pretty fitting.
    3. sum linear function and curve function. ->Y=w2(x^2) +f(x)
    4. determine curve coefficient by estimated RSS.
    5. go step 2(loop), to repeat as much as stepsize.
    6. stop if delta-RSS if smaller than 1e-4
    7. sum function with linear one.
  - ![program output](https://github.com/ibo52/ann-ml/blob/master/out%20imgs/polinomial_example.gif)
=======
>>>>>>> 8a4ef523ec5ffea4869786b832f4cbc46e95fbf6
