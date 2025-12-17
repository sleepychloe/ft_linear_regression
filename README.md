Mandatory part + every Bonus part

Tested on Linux

Success 125/100

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

#### Multivariable (nD, vectorized) <a name="cost-function-nd"></a>

```
	J(θ) = 1/2m ⋅ ∥Xθ - y∥²
	     = 1/2m ⋅ (Xθ - y)ᵀ ⋅ (Xθ - y)

	Where X ∈ Rᵐˣⁿ: design matrix
	      θ ∈ Rⁿˣ¹
	      y ∈ Rᵐˣ¹
```
<br>

### Gradient Descent <a name="linear-regression-gradient-descent"></a>

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

Proof:

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
<br>

#### Multivariable (nD, vector & matrix) <a name="gradient_descent-nd"></a>

```
	θ := θ - α ⋅ 1/m ⋅ Xᵀ ⋅ (Xθ - y)

	Where α is the learning rate
```
<br>
This works for any number of features,<br>
including single variable form (as you make form of x as design matrix).<br>
<br>

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
<br>


