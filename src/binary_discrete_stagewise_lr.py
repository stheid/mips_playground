"""
This implementation is basically the binary version of the MSL.
Unlike PSL, there is no mapping function from scores to probabilities.
Instead, each stage can be interpret as a logistic regression.
All stages share the weights for the features they share.

Unlike the discrete LR setting, we do not only have one loss, but a loss for each stage.
The loss is the sum of the losses of all stages.
The constraints are therefore also distributed to the different loss variables depending on the number of model variables involved.
E.g. for stage 1 we add loss constraints for each single feature.
The selected features for the last stage will therefore contain the scores for all previous stages.
"""

from tqdm import tqdm
from mip import Model, xsum, INTEGER, minimize, BINARY
import jax.numpy as jnp
import numpy as np
from jax import grad
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import optax
from itertools import combinations


def prepare():
    # --- MIP Model ---
    model = Model()


    losses = dict()
    selector = dict()
    for i in range(d+1):
        model.add_var(var_type=INTEGER, lb=-10, ub=10, name=f"w{i}")

        subselect = []
        for subset in combinations(range(d+1), i):
            losses[subset] = model.add_var(lb=0, name=f"loss{subset}")
            if 0 < i < d:
                select = model.add_var(var_type=BINARY, name=f"sel{subset}")
                selector[subset] = select
                subselect.append(select)

        if 0 < i < d:
            model.add_constr(xsum(subselect) == 1, f"select_one_loss_{i}")
       

    model.objective = minimize(
        losses[()]+ xsum(losses[i]*selector[i] for i in selector.keys())+ losses[tuple(range(d+1))]  # Add the loss for the last stage
    )  # Add the integer variables to the objective
    return model


if __name__ == "__main__":
    # load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    X = X[y != 2]  # binary classification
    y = y[y != 2]  # binary classification
    n, d = X.shape

    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

    # --- JAX loss & gradient ---
    def logloss(w, X, y):
        return optax.sigmoid_binary_cross_entropy(X @ w.T, y).mean() + 1e-5 * jnp.sum(
            w**2
        )

    logloss_grad = grad(logloss)

    model = prepare()
    model.verbose = 0

    for _ in tqdm(range(10)):
        model.optimize()
        w_vars = list(model.vars)[d+1:]
        w = [var.x for var in w_vars]
        w_np = np.array(w)

        # for each subset of features
        for i in range(d+1):
            stage_loss = model.vars[f"loss{i}"].x
            for subset in combinations(range(d+1), i):
                w_np_tmp = w_np.copy()
                w_np_tmp[list(subset)] = 0
                w_jnp = jnp.array(w_np_tmp)
 
                # add cut
                f = float(logloss(w_jnp, X_bias, y))
                g = [float(val) for val in logloss_grad(w_jnp, X_bias, y)]
                model += model.vars[f"loss{i}"] >= f + xsum(
                    g[j] * (w_vars[j] - w[j]) for j in range(d + 1)
                )
        print( model.objective_value, f)

    # --- Output ---
    print("Learned integer weights:", [v.x for v in model.vars[d+1:]])
    print("Objective value:", model.objective_value, f)
    print()

    lr = LogisticRegression(max_iter=100000, penalty=None)
    lr.fit(X, y)
    print("Learned float weights:", lr.coef_[0])
    print("Learned float bias:", lr.intercept_[0])
    lr_w = np.hstack(([lr.intercept_[0]], lr.coef_[0]))
    print("Reference performance:", logloss(lr_w, X_bias, y))
