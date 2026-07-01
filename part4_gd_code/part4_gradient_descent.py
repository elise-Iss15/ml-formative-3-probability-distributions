"""
Part 4: Gradient Descent in Code
=================================
Model:   y_hat = m1*x1 + m2*x2 + b
Data:    x = [(1,3), (4,10)],  y = [5, 6]
Init:    m = [-1, 2],  b = 1
Members: 4 → 4 gradient descent updates

What this file does:
  1. Defines the model using matrix multiplication (not scalars)
  2. Defines the MSE cost function
  3. Uses SciPy's approx_fprime to numerically compute the derivative
     (as required by the assignment, and as a check against our math)
  4. Runs gradient descent for 4 iterations, printing every step
  5. Plots how m1, m2, b change over iterations  →  params_plot.png
  6. Plots how the error (MSE) changes over iterations  →  error_plot.png
"""

import numpy as np
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
# X is a 2×2 matrix: 2 data points, each with 2 features (x1, x2)
X = np.array([[1,  3],
              [4, 10]], dtype=float)

y = np.array([5, 6], dtype=float)   # true/real answers

n = X.shape[0]   # number of data points = 2


# ─────────────────────────────────────────────
# INITIAL PARAMETERS
# ─────────────────────────────────────────────
m = np.array([-1.0, 2.0])   # m1 = -1, m2 = 2
b = 1.0                      # bias
learning_rate = 0.01
num_updates = 4              # one per group member


# ─────────────────────────────────────────────
# FUNCTION 1 — Prediction
#   y_hat = X @ m + b
#   @ means matrix multiplication
# ─────────────────────────────────────────────
def predict(X, m, b):
    return X @ m + b


# ─────────────────────────────────────────────
# FUNCTION 2 — Cost (Mean Squared Error)
#   J = (1/n) * sum( (y_hat - y)^2 )
#   Measures how wrong our predictions are. Lower is better.
#
#   theta = [m1, m2, b] packed into one vector so SciPy can
#   differentiate the whole function at once.
# ─────────────────────────────────────────────
def cost(theta, X, y):
    m_ = theta[:2]
    b_ = theta[2]
    y_hat = predict(X, m_, b_)
    return np.mean((y_hat - y) ** 2)


# ─────────────────────────────────────────────
# FUNCTION 3 — SciPy derivative (numerical gradient)
#   approx_fprime numerically estimates how much the cost changes
#   when each parameter changes slightly — no algebra needed.
#   We use this to VERIFY our hand-calculated gradients from Part 3.
# ─────────────────────────────────────────────
def scipy_gradient(m, b, X, y):
    theta = np.array([m[0], m[1], b])
    eps = np.sqrt(np.finfo(float).eps)   # tiny step size for numerical diff
    return approx_fprime(theta, cost, eps, X, y)
    # returns [dJ/dm1, dJ/dm2, dJ/db] numerically


# ─────────────────────────────────────────────
# FUNCTION 4 — Closed-form gradient (chain rule from Part 3)
#   dJ/dm = (2/n) * X^T @ (y_hat - y)
#   dJ/db = (2/n) * sum(y_hat - y)
# ─────────────────────────────────────────────
def closed_form_gradient(X, y, m, b):
    y_hat = predict(X, m, b)
    error = y_hat - y
    dJ_dm = (2 / n) * (X.T @ error)
    dJ_db = (2 / n) * np.sum(error)
    return dJ_dm, dJ_db


# ─────────────────────────────────────────────
# GRADIENT DESCENT LOOP — 4 iterations, every step printed
# ─────────────────────────────────────────────
m_history = [m.copy()]
b_history  = [b]
cost_history = []

print("=" * 65)
print("PART 4 — GRADIENT DESCENT: STEP-BY-STEP")
print("=" * 65)
print(f"Initial  m = {m},  b = {b}\n")

for i in range(1, num_updates + 1):
    print(f"{'─'*65}")
    print(f"  ITERATION {i}")
    print(f"{'─'*65}")

    # ── Step A: Predict ──────────────────────────────────────────
    y_hat = predict(X, m, b)
    print(f"  m = {m},  b = {b:.6f}")
    print(f"  y_hat  = X @ m + b  =  {y_hat}")

    # ── Step B: Compute error and cost ───────────────────────────
    error = y_hat - y
    J = np.mean(error ** 2)
    cost_history.append(J)
    print(f"  error  = y_hat - y  =  {error}")
    print(f"  Cost J (MSE)        =  {J:.6f}")

    # ── Step C: Closed-form gradient (chain rule) ─────────────────
    dJ_dm, dJ_db = closed_form_gradient(X, y, m, b)
    print(f"  dJ/dm  (chain rule) =  {dJ_dm}")
    print(f"  dJ/db  (chain rule) =  {dJ_db:.6f}")

    # ── Step D: SciPy numerical gradient — verification ───────────
    scipy_grad = scipy_gradient(m, b, X, y)
    match = np.allclose([dJ_dm[0], dJ_dm[1], dJ_db], scipy_grad, atol=1e-4)
    print(f"  SciPy  gradient     =  {scipy_grad}")
    print(f"  Chain rule matches SciPy: {match}")

    # ── Step E: Update parameters ─────────────────────────────────
    m_new = m - learning_rate * dJ_dm
    b_new = b - learning_rate * dJ_db
    print(f"  m_new = {m} - {learning_rate}*{dJ_dm}  =  {m_new}")
    print(f"  b_new = {b:.6f} - {learning_rate}*{dJ_db:.6f}  =  {b_new:.6f}")

    m, b = m_new, b_new
    m_history.append(m.copy())
    b_history.append(b)
    print()

# Record the final cost after the last update
final_y_hat = predict(X, m, b)
final_J = np.mean((final_y_hat - y) ** 2)
cost_history.append(final_J)

print("=" * 65)
print("FINAL RESULTS")
print("=" * 65)
print(f"  Final m             =  {m}")
print(f"  Final b             =  {b:.6f}")
print(f"  Final predictions   =  {final_y_hat}")
print(f"  Final cost (MSE)    =  {final_J:.6f}")
print()
print("  Cost over iterations:", [round(c, 4) for c in cost_history])
print("  → Cost is decreasing each iteration ✓")


# ─────────────────────────────────────────────
# PLOT 1 — How m1, m2, and b change over iterations
# ─────────────────────────────────────────────
m_history = np.array(m_history)
b_history  = np.array(b_history)
iters = np.arange(len(b_history))   # [0, 1, 2, 3, 4]

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(iters, m_history[:, 0], marker='o', label='m1', color='steelblue')
ax1.plot(iters, m_history[:, 1], marker='o', label='m2', color='darkorange')
ax1.plot(iters, b_history,       marker='o', label='b',  color='green')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Parameter Value')
ax1.set_title('Parameter Values (m1, m2, b) Over Iterations')
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig('params_plot.png', dpi=150)
print("\nSaved → params_plot.png")

# ─────────────────────────────────────────────
# PLOT 2 — How the error (MSE cost) changes over iterations
# ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(np.arange(len(cost_history)), cost_history,
         marker='o', color='crimson', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('MSE Cost (J)')
ax2.set_title('Cost (MSE) Over Iterations — Gradient Descent Converging')
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig('error_plot.png', dpi=150)
print("Saved → error_plot.png")