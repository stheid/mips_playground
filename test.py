from tqdm   import tqdm 
from mip import Model, xsum, INTEGER, minimize, ConstrsGenerator
import jax.numpy as jnp
from jax import grad
from sklearn.datasets import make_classification

# --- Prepare dataset ---
X_raw, y_raw = make_classification(n_samples=20, n_features=5, random_state=2)
X = jnp.array(X_raw)
y = jnp.array(2 * y_raw - 1)  # Convert {0,1} -> {-1,1}
n, d = X.shape

# --- JAX loss & gradient ---
def logloss(w, x, y):
    return jnp.log(1 + jnp.exp(-y * jnp.dot(x, w)))

logloss_grad = grad(logloss)

# --- MIP Model ---
model = Model()
w_vars = [model.add_var(var_type=INTEGER, lb=-10, ub=10, name=f"w{i}") for i in range(d)]
l_vars = [model.add_var(lb=0, name=f"l{i}") for i in range(n)]

model.objective = minimize(xsum(l_vars))

initial_w = jnp.zeros(d)
initial_w_np = [float(val) for val in initial_w]  # convert to Python floats

for i in range(n):
    g = logloss_grad(initial_w, X[i], y[i])
    f = logloss(initial_w, X[i], y[i])

    g_np = [float(val) for val in g]
    f_np = float(f)

    tangent = xsum(g_np[j] * (w_vars[j] - initial_w_np[j]) for j in range(d)) + f_np
    model += (l_vars[i] >= tangent)

# --- Cuts Generator using JAX ---
class LogLossLatticeCuts(ConstrsGenerator):
    def generate_constrs(self, m,depth: int, pass_number: int):
        try:
            violated = 0
            for i in range(n):
                # Get current solution
                w_curr = jnp.array([var.x for var in list(m.vars)[:d]])                
                g = logloss_grad(w_curr, X[i], y[i])
                f = logloss(w_curr, X[i], y[i])
                w_curr_np = [float(val) for val in w_curr]
                g_np = [float(val) for val in g]
                f_np = float(f)
                
                # Lattice cuts: one cut above and one below
                upper_tangent = xsum(float(g_np[j]) * (w_vars[j] - w_curr_np[j]) for j in range(d)) + float(f_np)
                lower_tangent = xsum(float(g_np[j]) * (w_vars[j] - w_curr_np[j]) for j in range(d)) + float(f_np) - 1e-4  # Slight offset
                
                # Check if the lower bound is violated (if the variable is lower than the tangent)
                if l_vars[i].x < float(f) - 1e-4:
                    m += (l_vars[i] >= upper_tangent)  # Add upper cut
                    m += (l_vars[i] >= lower_tangent)  # Add lower cut
                    violated += 1
                    
            return violated > 0  # Return whether cuts were added
        
        except Exception as e:
            import traceback
            print("Error in generate_constrs:", e)
            traceback.print_exc()


# --- Attach and solve ---
model.cuts_generator = LogLossLatticeCuts()
model.verbose = 1
model.optimize()

# --- Output ---
w_sol = [int(var.x) for var in w_vars]
print("Learned integer weights:", w_sol)
