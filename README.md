Mandatory part + every Bonus part

Tested on Linux

Success 125/100


## Lists
 * [Demo](#demo) <br>
 * [Project ft_linear_regression](#project-ft-linear_regression) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Mandatory Part](#project-mandatory) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Generalization Beyond the Subject](#project-beyond-mandatory) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Bonus Part](#project-bonus) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Standard Regression Metrics](#project-bonus-1) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Early Stopping via Cost Convergence](#project-bonus-2) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Installation](#installation) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Usage](#usage) <br>
 * [Linear Regression](#linear-regression) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Linear Regression](#linear-regression-linear-regression) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Single variable (1D)](#linear-regression-1d) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Multivariable (nD)](#linear-regression-nd) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Problem Formulation](#linear-regression-problem-formulation) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Cost Function (MSE, Mean Squared Error)](#linear-regression-cost-function) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Single variable (1D)](#cost-function-1d) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Multivariable (nD)](#cost-function-nd) <br>
 * [Gradient Descent](#gradient-descent) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Gradient Descent](#gradient-descent-gradient-descent) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Single variable (1D)](#gradient-descent-1d) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Multivariable (nD)](#gradient-descent-nd) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Learning Rate α](#gradient-descent-learning-rate) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Feature Normalization](#gradient-descent-normalization) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Single variable (1D)](#normalize-1d) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⋅ [Multivariable (nD)](#normalize-nd) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [Convexity (Linear Regression guarantees a global minimum)](#gradient-descent-convexity) <br>

<br>

## Demo <a name="demo"></a>
![Animated GIF](https://github.com/sleepychloe/ft_linear_regression/blob/main/img/iter_250.gif)

###### ↳ iteration ≈ 250

<br>

![Animated GIF](https://github.com/sleepychloe/ft_linear_regression/blob/main/img/iter_1000.gif)

###### ↳ iteration = 1000 (max iter)

<br>
<br>
<br>

## Project ft_linear_regression <a name="project-ft-linear-regression"></a>

### Mandatory Part <a name="project-mandatory"></a>

This project implements a simple linear regression model with a single feature, as required by the subject.<br>

<br>

1. Training Program

- Reads a dataset containing `mileage`(input feature) and `price`(target value)

- Trains a linear regression model using Gradient Descent

- Initializes parameters as:
```
	θ₀ = 0
	θ₁ = 0
```

- Iteratively updates parameters using:
```
	θ₀ := θ₀ - α ⋅ 1/m ⋅ ∑ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)
	θ₁ := θ₁ - α ⋅ 1/m ⋅ ∑ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) ⋅ x⁽ⁱ⁾
```

- Saves the learned parameters(θ₀, θ₁) for later use

<br>

2. Prediction Program

- Loads the trained parameters θ₀ and θ₁

- Prompts the user for a mileage value

- Predicts the price using:
```
	estimatePrice(mileage) = θ₀ + θ₁ ⋅ mileage
```
<br>
<br>

#### Generalization Beyond the Subject <a name="project-beyond-mandatory"></a>

Although the subject only requires simple linear regression with a single feature<br>
I implemented a generalized linear regression class that supports multiple variables.<br>
<br>
This design allows the same implementations to be reused for:<br>

- single-variable regression

- multivariable regression

- future extensions


```
class LinearRegression:
	def __init__(self):
		...
		self.theta: Vector | None = None
		...
	...
	def hypothesis(self, X: Matrix) -> Vector:
		if self.theta is None:
			raise RuntimeError("Model is not fitted yet")
		return X @ self.theta

	def mse_cost(self, X: Matrix, y: Vector) -> float:
		m = len(y)
		errors = self.hypothesis(X) - y
		return (1 / (2 * m)) * np.sum(errors ** 2)

	def gradient_step(self, X: Matrix, y: Vector) -> None:
		m = len(y)
		errors = self.hypothesis(X) - y
		gradient = (1 / m) * (X.T @ errors)
		self.theta -= self.learning_rate * gradient
	...
```

<br>
To use this generalized implementation for a single feature,<br>
following transformations are applied:<br>

1. Convert the input feature X into a design matrix

2. Represent parameters θ as a column vector


```
	X_norm = model.normalize_X(x.reshape(-1, 1))
	model.theta = np.zeros(X_norm.shape[1])
```

This keeps the mathematical formulation consistent while while respecting subject requirements.<br>

<br>
<br>

### Bonus Part <a name="project-bonus"></a>

All bonus features described in the subject have been implemented:<br>

- Scatter plot of the dataset to visualize data distribution

- Regression line plotted on the same graph

- The accuracy of the trained model


<br>
<br>

#### Standard Regression Metrics <a name="project-bonus-1"></a>

I have evaluated using 4 standard regression metrics:<br>

1. MSE (Mean Squared Error)

: Penalize large errors more heavily<br>
  Directly corresponds to the optimized cost function<br>

```
	MSE = 1/m ⋅ ∑ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
```

2. RMSE (Root Mean Squared Error)

: Same unit as the target variable<br>
  Easier to interpret than MSE<br>

```
	RMSE = √MSE
```

3. MAE (Mean Absolute Error)

: Measures avery absolute deviation<br>
  Less sensitive to outliers than MSE<br>

```
	MAE = 1/m ⋅ ∑ | ŷ⁽ⁱ⁾ - y⁽ⁱ⁾ |
```

4. Coefficient of Determination (R²)

: Measures how well the model explains variance in the data<br>
  R² ≈ 1 indicates a string linear fit<br>

```
	            1/m ⋅ ∑ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
	R² = 1 - ( ──────────────────────── )
	           1/m ⋅ ∑ (y⁽ⁱ⁾ - y_mean)
```

<br>
These metrices provide complementary perspectives on model performance.<br>


<br>
<br>

#### Early Stopping via Cost Convergence <a name="project-bonus-2"></a>

According to the subject, the training program only requires updating θ₀ and θ₁<br>
using Gradient Descent for a fixed number of iterations.<br>
In principle, the cost function does not need to be explicitly monitored.<br>
<br>
However, in this project, I introduced the cost function (MSE)<br>
to detect convergence and enable early stopping,<br>
making the training both more efficient and more numerically robust.<br>

<br>
<br>
Gradient Descent is an iterative algorithm that converges asymptotically toward the minimum.<br>
After a certain number of iterations, parameter updates become negligibly small,<br>
and further iterations do not significantly improve the model.<br>
<br>
Running the algorithm beyond this point:<br>

- Increases computation time unnecessarily
- Does not meaningfully improve prediction accuracy
- May accumulate floating-point noise

To address this, I added a convergence-based stopping criterion.<br>
<br>

- for each iteration, the corrent cost is computed
- training stopes early if the improvement between cost and previous cost is smaller than ε
- additional safety checks are included to detect divergence

→ This means that Gradient Descent terminates as soon as it reaches a numerically stable minimum, instead of blindly iterating up to MAX_ITER.<br>

```
	for i in range(model.max_iter):
		model.gradient_step(X_norm, y)
		cost = model.mse_cost(X_norm, y)
		if abs(prev_cost - cost) < model.epsilon:
			break
		elif np.isnan(cost) or cost > 1e12:
			print("Diverged")
			break
		...
		prev_cost = cost
```

##### Experimental Observation

I compared two cases:

1. early stopping enabled: training stops when |Δcost| < EPSILON(=1e-5)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Convergence occurs around 250–260 iterations

2. training runs until MAX_ITER(=1000)

<br>

<img src="https://github.com/sleepychloe/ft_linear_regression/blob/main/img/iter_250_precision.png" width="485" height="75">

###### ↳ iteration ≈ 250

<img src="https://github.com/sleepychloe/ft_linear_regression/blob/main/img/iter_1000_precision.png" width="485" height="75">

###### ↳ iteration = 1000 (max iter)

<br>
The resulting accuracy metrics are nearly identical.<br>
<br>
This demonstrates that most of the learning happens in the early phase,<br>
and that continuing to iterate after convergence provides no practical benefit.<br>

<br>
<br>
<br>

## Installation & Usage <a name="installation-usage"></a>

### Installation <a name="installation"></a>

```
	git clone https://github.com/sleepychloe/ft_linear_regression.git
	cd ft_linear_regression
```

If docker and docker-compose is not isntalled, install it via
```
	sudo apt install -y docker.io && sudo apt-get install -y docker-compose
```

<br>
<br>

### Usage <a name="usage"></a>

To run,
```
	cd ft_linear_regression
	make
	(url) http://127.0.0.1:8888
```

<br>

To see lists of containers, volumes, images, and networks,
```
   make list
```

To see outputs of containers,
```
   make logs
```

To stop containers,
```
   make stop
```

To restart containers,
```
   make restart
```

To clean every containers, volumes, images, and networks,
```
   make fclean
```

<br>
<br>
<br>

## Linear Regression <a name="linear-regression"></a>
### Linear Regression <a name="linear-regression-linear-regression"></a>

Linear Regression is a method used to model the relationship<br>
between `Input variable(features)` and `Output variable(target)`<br>
by assuming a linear relationship between them.<br>
<br>

#### Single variable (1D) <a name="linear-regression-1d"></a>

```
	ŷ = θ₀ + θ₁x

	x: input feature
	ŷ: predicted output
	θ₀: bias(intercept)
	θ₁: weight(slope)
```
<br>

#### Multivariable (nD) <a name="linear-regression-nd"></a>

```
	ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ⋯ + θₙxₙ
	  = Xθ
```
<br>
<br>

### Problem Formulation <a name="linear-regression-problem-formulation"></a>

We are given a dataset:<br>

```
	             ₘ
	{(x⁽ⁱ⁾,y⁽ⁱ⁾)}
	             ⁱ⁼¹
```
<br>
Our goal is to find parameters θ such that<br>
the predicted values ŷ⁽ⁱ⁾ are as close as possible to the true values y⁽ⁱ⁾.<br>
<br>
<br>

### Cost Function (MSE, Mean Squared Error) <a name="linear-regression-cost-function"></a>

To quantify how wrong our predictions are,<br>
we define a cost function.<br>
<br>

#### Single variable (1D) <a name="cost-function-1d"></a>

```
	                   ₘ
	J(θ₀, θ₁) = 1/2m ⋅ ∑ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
	                  ⁱ⁼¹
```
<br>

#### Multivariable (nD) <a name="cost-function-nd"></a>

```
	J(θ) = 1/2m ⋅ ∥Xθ - y∥²
	     = 1/2m ⋅ (Xθ - y)ᵀ ⋅ (Xθ - y)

	Where X ∈ Rᵐˣⁿ: design matrix
	      θ ∈ Rⁿˣ¹
	      y ∈ Rᵐˣ¹
```
<br>
<br>
<br>

## Gradient Descent <a name="gradient-descent"></a>
### Gradient Descent <a name="gradient-descent-gradient-descent"></a>

Gradient Descent is an iterative optimization algorithm<br>
used to minimize the cost function.<br>
At each step, we update parameters in the opposite direction(-) of the gradient.<br>
<br>

#### Single variable (1D) <a name="gradient-descent-1d"></a>

```
	               ∂J
	θ₀ := θ₀ - α ⋅ ───
	               ∂θ₀
	    = θ₀ - α ⋅ 1/m ⋅ ∑ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)

	               ∂J
	θ₁ := θ₁ - α ⋅ ───
	               ∂θ₁
	    = θ₁ - α ⋅ 1/m ⋅ ∑ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) ⋅ x⁽ⁱ⁾
	
	Where α is the learning rate
```
<br>

<details>
<summary><b><ins>Proof</ins></b></summary>


```
Starting from:

	J = 1/2m ⋅ ∑ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
	  = 1/2m ⋅ ∑ ((θ₀ + θ₁x⁽ⁱ⁾) - y⁽ⁱ⁾)²

The goal is to find θ that minimizes the cost (=the gradient is 0).
Also, linear regression always guarantees a global minimum.
→ Compute the gradient(do partial differentiate) with respect to each theta
(Why linear regression guarantees a global minimum will be explained later)

Derivative wrt θ₀:

	 ∂
	─── ( 1/2m ⋅ ∑ ((θ₀ + θ₁x⁽ⁱ⁾) - y⁽ⁱ⁾)² )
	∂θ₀

	= 1/2m ⋅ 2 ⋅ ∑ ((θ₀ + θ₁x⁽ⁱ⁾) - y⁽ⁱ⁾) ⋅ 1

	= 1/m ⋅ ∑ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)

Derivative wrt θ₁:

	 ∂
	─── ( 1/2m ⋅ ∑ ((θ₀ + θ₁x⁽ⁱ⁾) - y⁽ⁱ⁾)² )
	∂θ₁

	= 1/2m ⋅ 2 ⋅ ∑ ((θ₀ + θ₁x⁽ⁱ⁾) - y⁽ⁱ⁾) ⋅ x⁽ⁱ⁾

	= 1/m ⋅ ∑ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) ⋅ x⁽ⁱ⁾
```

</details>
<br>

#### Multivariable (nD) <a name="gradient-descent-nd"></a>

```
	θ := θ - α ⋅ ∇J(θ)
	   = θ - α ⋅ 1/m ⋅ Xᵀ ⋅ (Xθ - y)

	Where α is the learning rate
```
<br>
This works for any number of features,<br>
including single variable form (as you make form of x as design matrix).<br>
<br>


<details>
<summary><b><ins>Proof</ins></b></summary>

Proof:

```
Hypothesis:
	ŷ = Xθ

Cost Function:
	J(θ) = 1/2m ⋅ ∥Xθ - y∥²
	     = 1/2m ⋅ (Xθ - y)ᵀ ⋅ (Xθ - y)

The goal is to find θ that minimizes the cost.
Also, linear regression always guarantees a global minimum.
→ Compute the gradient of J(θ) with respect to θ
   to find the minimizer θ′ satisfies ∇J(θ′) = 0
(And because J is convex, this stationary point is the global minimum,
which will be explained later)

Gradient:

	   ∂
	= ── ( J(θ) )
	  ∂θ

	   ∂
	= ── ( 1/2m ⋅ (Xθ - y)ᵀ ⋅ (Xθ - y) )
	  ∂θ

	   ∂
	= ── ( 1/2m ⋅ (θᵀXᵀXθ - 2θᵀXᵀy + yᵀy) )      ⋯ ⓐ
	  ∂θ

	= 1/2m ⋅ (2XᵀXθ - 2Xᵀy + 0)
	= 1/m ⋅ (XᵀXθ - Xᵀy)
	= 1/m ⋅ Xᵀ ⋅ (Xθ - y)



** note **
            ∂
in the ⓐ : ── ( θᵀXᵀXθ - 2θᵀXᵀy + yᵀy),
            ∂θ

                             ∂
    1. when A is symmetric, ── (θᵀAθ) = 2Aθ
                            ∂θ
                          ∂
        → the first term ── (θᵀXᵀXθ) = 2XᵀXθ
                         ∂θ
          (XᵀX is symmetric, so the identity applies)

        ∂
    2. ── (bᵀθ) = b
       ∂θ
                           ∂
        → the second term ── (2θᵀXᵀy) = 2Xᵀy
                          ∂θ

    3. the third term yᵀy is independent of θ
        → vanishes when differentiated
```

</details>
<br>
<br>

### Learning Rate α <a name="gradient-descent-learning-rate"></a>

```
	θ := θ - α ⋅ ∇J(θ)
	   = θ - α ⋅ 1/m ⋅ Xᵀ ⋅ (Xθ - y)

	Where α is the learning rate
```

learning rate α: controls how far we move in parameter space per iteration.<br>

- when α is too large: the algorithm may diverge
- when α is too small: convergence becomes extremely slow

Even though linear regression has a unique global minimum(convex),<br>
Gradient Descent can still fail numerically if α is poorly chosen.<br>

<br>
<br>

### Feature Normalization <a name="gradient-descent-normalization"></a>

Gradient Descent is sensitive to feature scaling.<br>
Ill-conditioned(highly elongated) cost surface may cause slow and unstable convergence.<br>
Normalizing features makes the optimization landscape more spherical,<br>
so a single learning rate α can work well across all parameters.<br>

- Z-score Normalization: A value expressed in standard deviation units that indicates how far a value in a data set deviates from the mean. (adjusts the mean to 0, standard deviation to 1)

<br>

#### Single variable (1D) <a name="normalize-1d"></a>


```
	μ = mean(x)
	σ = standard deviation(x)
```

- Normalize x:

```
	          x - μ
	x_norm = ───────
	            σ
```
<br>

- Unnormalize x:

```
	x = x_norm ⋅ σ + μ
``` 
<br>

#### Multivariable (nD) <a name="normalize-nd"></a>


```
	μ ∈ Rⁿ where μⱼ = mean of column j
	σ ∈ Rⁿ where σⱼ = standard deviation of column j
```

- Normalize X:

```
	             Xᵢⱼ - μⱼ
	(X_norm)ᵢⱼ = ────────
	                σⱼ
	
	→ X_norm = (X - 1⋅μᵀ) ⊘ (1⋅σᵀ)

	1 ∈ Rᵐ: a column vector of ones
	⊘: elementwise division
```

<br>

- Unnormalize X:

```
	Xᵢⱼ = (X_norm)ᵢⱼ ⋅ σⱼ + μⱼ

	→ X = (X_norm ⊙ (1⋅σᵀ)) + (1⋅μᵀ)

	⊙: elementwise multiplication
```

<br>

<details>
<summary><b><ins>The reason why form of `X - 1 ⋅ μᵀ`, not `X - μᵀ`</ins></b></summary>


```
Assumption:
        X ∈ ℝᵐˣⁿ: design matrix
        m: number of samples
        n: number of features
        each column represents a feature
        normalization must be performed on a column-by-column basis

Thus,
        μ ∈ ℝⁿ: the mean of each feature → Vector(size: n)
        X ∈ ℝᵐˣⁿ → Matrix(size: m*n)
```
→ cannot compute n-vector with m*n matrix


```
the meaning of 1 ⋅ μᵀ:
        1 ∈ ℝᵐ: a column vector in which elements are all 1
        μᵀ ∈ ℝ¹ˣⁿ

Thus,
        1 ⋅ μᵀ ∈ Rᵐˣⁿ
               ┏            ┓
               ┃ μ₁ μ₂ … μₙ ┃
        1⋅μᵀ = ┃ μ₁ μ₂ … μₙ ┃
               ┃ ⋮  ⋮    ⋮  ┃
               ┃ μ₁ μ₂ … μₙ ┃
               ┗            ┛
```
→ X - 1 ⋅ μᵀ ∈ ℝᵐˣⁿ

</details>

- Unnormalize θ:

```
	θ′₀ = θ₀ - (θ′)ᵀ ⋅ μ
	θ′ = θ ⊘ σ
```

<br>

<details>
<summary><b><ins>Proof</ins></b></summary>

```
	x_norm = (x - μ) ⊘ σ

	ŷ = θ₀ + θᵀ ⋅ x_norm
	  = θ₀ + θᵀ ⋅ ( (x - μ) ⊘ σ )
	  = θ₀ + θᵀ ⋅ (x ⊘ σ) - θᵀ ⋅ (μ ⊘ σ) 
	  = (θ₀ - θᵀ ⋅ (μ ⊘ σ)) + (θᵀ ⋅ (x ⊘ σ))
	  = (θ₀ - (θ ⊘ σ)ᵀ ⋅ μ) + (θ ⊘ σ)ᵀ ⋅ x
	    ╚═════════════════╝   ╚══════╝
	            θ′₀             (θ′)ᵀ

	∴ θ′ = θ ⊘ σ
	  θ′₀ = θ₀ - (θ ⊘ σ)ᵀ ⋅ μ
	      = θ₀ - (θ′)ᵀ ⋅ μ
```

</details>

<br>

### Convexity (Linear Regression guarantees a global minimum) <a name="gradient-descent-convexity"></a>

To explain why linear regression has no local minima<br>
we show that its cost function J(θ) is convex (bowl-shaped).<br>

A sufficient condition:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If the Hessian matrix ∇²J(θ) is Positive Semi-Definite(PSD),<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;then J is convex<br>


```
the cost function J(θ)
        = 1/2m ⋅ (Xθ - y)ᵀ ⋅ (Xθ - y)

1. Gradient ∇J(θ)
        = 1/m ⋅ Xᵀ ⋅ (Xθ - y)
        = 1/m ⋅ (XᵀXθ - Xᵀy)

2. Hessian ∇²J(θ)
        = 1/m ⋅ XᵀX

3. PSD
        (for any vector z)
        zᵀ ⋅ XᵀX ⋅ z
         = (zX)ᵀ ⋅ (zX)
         = ∥ zX ∥² ≥ 0

4. Conclusion
        XᵀX is PSD ⇒ ∇²J(θ) is PSD ⇒ J(θ) is convex,
        there are no local minima.
        Any stationary point (where ∇J(θ)=0) is a global minimum
```
<br>
<br>
