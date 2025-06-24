from tqdm import tqdm
from mip import Model, xsum, INTEGER, minimize, ConstrsGenerator
import numpy as np
from jax import grad
import jax.numpy as jnp
import optax
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


class LossCPAGenerator(ConstrsGenerator):
    def __init__(self, X, y, pbar):
        self.X = X
        self.y = y
        self.pbar = pbar

    def generate_constrs(self, model: Model, depth: int = 0, npass: int = 0):
        self.pbar.update()
        w_vars = list(model.vars)[1:]
        w = [var.x for var in w_vars]
        w_np = jnp.array(w)

        f = float(logloss(w_np, X_bias, y))
        g = [float(val) for val in logloss_grad(w_np, X_bias, y)]

        model += model.vars["loss"] >= f + xsum(
            g[j] * (w_vars[j] - w[j]) for j in range(d + 1)
        )


def logloss(w, X, y):
    return optax.sigmoid_binary_cross_entropy(X @ w.T, y).mean() + 1e-5 * jnp.sum(w**2)


logloss_grad = grad(logloss)


if __name__ == "__main__":
    # load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    X = X[y != 2]  # binary classification
    y = y[y != 2]  # binary classification
    n, d = X.shape
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

    model = Model()
    loss = model.add_var(lb=0, name="loss")
    for i in range(d + 1):
        model.add_var(var_type=INTEGER, lb=-10, ub=10, name=f"w{i}")
    model += loss >= 0
    model.objective = minimize(loss)  # Add the integer variables to the objective
    model.verbose = 0

    cut_gen = LossCPAGenerator(X_bias, y, tqdm())
    # create cuts for fractional solutions
    model.cuts_generator = cut_gen
    # create cuts for infeasible solutions
    model.lazy_constrs_generator = cut_gen
    model.optimize()

    print("Learned integer weights:", [v.x for v in model.vars[1:]])
    print("Objective value:", model.objective_value)
    print()

    lr = LogisticRegression(max_iter=100000, penalty=None)
    lr.fit(X, y)
    print("Learned float weights:", lr.coef_[0])
    print("Learned float bias:", lr.intercept_[0])
    lr_w = np.hstack(([lr.intercept_[0]], lr.coef_[0]))
    print("Reference performance:", logloss(lr_w, X_bias, y))
