from tqdm import tqdm 
from mip import Model, xsum, INTEGER, minimize
import jax.numpy as jnp
import numpy as np
from jax import grad
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import optax

"""
This is basically a reimlementation of risk-slim.

It fits a logistic regression model with integer weights.
The execution iteratively solves a MIP model and adds constraints to the model.
The cuts are generated from the exact loss function and 
    its gradient while the MIP solves the approximate loss function.
"""


def prepare():
    # --- MIP Model ---
    model = Model()
    loss = model.add_var(lb=0, name="loss")
    for i in range(d+1):
        model.add_var(var_type=INTEGER, lb=-10, ub=10, name=f"w{i}")

    model.objective = minimize(loss)  # Add the integer variables to the objective
    return model


def logloss(w, X, y):
    return optax.sigmoid_binary_cross_entropy(X@w.T, y).mean() + 1e-5 * jnp.sum(w**2)
logloss_grad = grad(logloss)


if __name__ == "__main__":
    # load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    X = X[y != 2]  # binary classification
    y = y[y != 2]  # binary classification
    n, d = X.shape

    X_bias = np.hstack((np.ones((X.shape[0],1)), X))

    model = prepare()
    model.verbose=0
    
    # while true with progress bar (iter is an infinite generator)
    for _ in tqdm(iter(int,1)):
        model.optimize()

        w_vars = model.vars[1:]
        w = [var.x for var in w_vars]
        w_np = jnp.array(w)
        prox_loss = model.vars["loss"].x

        # add cut
        f = float(logloss(w_np, X_bias, y))
        g = [float(val) for val in logloss_grad(w_np, X_bias, y)]
        model +=  model.vars["loss"]>= f + xsum(g[j] * (w_vars[j]-w[j]) for j in range(d+1))
        opt_gap = 1-prox_loss / f
        if opt_gap < 1e-10:
            break


    # --- Output ---
    print("Learned integer weights:", [v.x for v in model.vars[1:]]) 
    print("Objective value:", model.objective_value, f)
    print("Optimality gap:", opt_gap)
    print()


    lr = LogisticRegression(max_iter=100000, penalty=None)
    lr.fit(X, y)
    print("Learned float weights:", lr.coef_[0])
    print("Learned float bias:", lr.intercept_[0])
    lr_w = np.hstack(([lr.intercept_[0]], lr.coef_[0]))
    print("Reference performance:", logloss(lr_w,X_bias, y)) 