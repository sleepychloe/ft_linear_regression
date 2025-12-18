Mandatory part + every Bonus part

Tested on Linux

Success 125/100


## Lists
 * [Demo](#demo) <br>
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
