"""
This implementation is basically the binary version of the MSL.
Unlike PSL, there is no mapping function from scores to probabilities.
Instead, each stage can be interpret as a logistic regression.
All stages share the weights for the features they share.

The feature permutation are provided as an input. it can be fitted using random forrest feature importance or by using the permutations from the greedy search MSL algorithm.
The MIP itself only fits the scores for the features in each stage (and class).

Unlike the discrete LR setting, we do not only have one loss, but a loss for each stage.
The loss is the sum of the losses of all stages.
The loss constraints are therefore also distributed among the stages.
"""

from tqdm import tqdm
from mip import Model, xsum, INTEGER, minimize
import jax.numpy as jnp
import numpy as np
from jax import grad
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import optax


def prepare():
    # --- MIP Model ---
    model = Model()

    losses = [model.add_var(lb=0, name=f"loss{i}") for i in range(d + 1)]
    [model.add_var(var_type=INTEGER, lb=-10, ub=10, name=f"w{i}") for i in range(d + 1)]

    model.objective = minimize(xsum(losses))
    return model


if __name__ == "__main__":
    # load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    X = X[y != 2]  # binary classification
    y = y[y != 2]  # binary classification
    n, d = X.shape

    # Feature permutation (for example, from a random forest feature importance)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_
    feature_order = np.argsort(feature_importances)[::-1]  # descending order
    X = X[:, feature_order]  # reorder features based on importance
    print("Feature order:", feature_order)

    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

    # --- JAX loss & gradient ---
    def logloss(w, X, y):
        return optax.sigmoid_binary_cross_entropy(X @ w.T, y).mean() + 1e-5 * jnp.sum(
            w**2
        )

    logloss_grad = grad(logloss)

    model = prepare()
    model.verbose = 0

    # while true with progress bar (iter is an infinite generator)
    for _ in tqdm(iter(int, 1)):
        model.optimize()
        w_vars = list(model.vars)[d + 1 :]
        w = [var.x for var in w_vars]
        w_np = np.array(w)
        loss = 0

        # for each subset of features
        for i in range(d + 1):
            stage_loss = model.vars[f"loss{i}"].x

            w_np_tmp = w_np.copy()
            w_np_tmp[i + 1 :] = 0
            w_jnp = jnp.array(w_np_tmp)

            # add cut
            f = float(logloss(w_jnp, X_bias, y))
            g = [float(val) for val in logloss_grad(w_jnp, X_bias, y)]
            model += model.vars[f"loss{i}"] >= f + xsum(
                g[j] * (w_vars[j] - w[j]) for j in range(i + 1)
            )

            loss += f

        opt_gap = 1 - model.objective_value / loss
        if opt_gap < 1e-10:
            break

    # --- Output ---
    print("Learned losses:", [v.x for v in model.vars[: d + 1]])
    print("Learned integer weights:", [v.x for v in model.vars[d + 1 :]])
    print("Objective value:", model.objective_value, f)
    print()

    lr = LogisticRegression(max_iter=100000, penalty=None)
    lr.fit(X, y)
    print("Learned float weights:", lr.coef_[0])
    print("Learned float bias:", lr.intercept_[0])
    lr_w = np.hstack(([lr.intercept_[0]], lr.coef_[0]))
    print("Reference performance:", logloss(lr_w, X_bias, y))
