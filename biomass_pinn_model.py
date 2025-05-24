
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import deepxde as dde
from deepxde.nn import FNN
import matplotlib.pyplot as plt

# === Constants ===
mu_max = 1.0
K_s = 0.5
Y = 0.5

# Initial conditions
S0 = 1.0
X0 = 0.1

# === Time domain ===
t_train = np.linspace(0, 1, 200)

# True growth rate function
def mu_t_func(t):
    return mu_max * np.sin(0.5 * t) ** 2

# === Solve ODE system ===
def monod_ode_system(t, y):
    S, X = y
    mu_t = mu_t_func(t)
    dS_dt = -1 / Y * mu_t * S / (K_s + S) * X
    dX_dt = mu_t * S / (K_s + S) * X
    return [dS_dt, dX_dt]

sol = solve_ivp(monod_ode_system, [t_train[0], t_train[-1]], [S0, X0], t_eval=t_train)
S = sol.y[0]
X = sol.y[1]
mu_values = mu_t_func(t_train)

# === Interpolation for true data ===
S_fn = interp1d(t_train, S, kind="cubic", fill_value="extrapolate")
X_fn = interp1d(t_train, X, kind="cubic", fill_value="extrapolate")
mu_fn = interp1d(t_train, mu_values, kind="cubic", fill_value="extrapolate")

def data_true(t):
    t = np.array(t).flatten()
    S_vals = S_fn(t)
    X_vals = X_fn(t)
    mu_vals = mu_fn(t)
    return np.hstack([
        S_vals.reshape(-1, 1),
        X_vals.reshape(-1, 1),
        mu_vals.reshape(-1, 1),
    ])

# === PINN model ODE residuals ===
def monod_ode(t, y):
    S, X, mu = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    dS_dt = dde.grad.jacobian(y, t, i=0)
    dX_dt = dde.grad.jacobian(y, t, i=1)
    mu_expr = mu * S / (K_s + S)
    eq1 = dS_dt + (1 / Y) * mu_expr * X
    eq2 = dX_dt - mu_expr * X
    return [eq1, eq2]

geom = dde.geometry.TimeDomain(0, 1)

ic1 = dde.IC(geom, lambda t: S0, lambda _, on_initial: on_initial, component=0)
ic2 = dde.IC(geom, lambda t: X0, lambda _, on_initial: on_initial, component=1)
ic3 = dde.IC(geom, lambda t: mu_t_func(t), lambda _, on_initial: on_initial, component=2)

data = dde.data.PDE(
    geom,
    monod_ode,
    [ic1, ic2, ic3],
    num_domain=200,
    solution=data_true,
    num_test=200,
)

layer_size = [1] + [64] * 6 + [3]
activation = "relu"
net = FNN(layer_size, activation=activation, kernel_initializer="Glorot uniform")
model = dde.Model(data, net)

model.compile("adam", lr=0.001 )
losshistory, train_state = model.train(epochs=10000)

# === Predictions and error ===
t_test = np.linspace(0, 1, 200).reshape(-1, 1)
y_pred = model.predict(t_test)
y_true = data_true(t_test)

mse_s = np.mean((y_pred[:, 0] - y_true[:, 0]) ** 2)
mse_x = np.mean((y_pred[:, 1] - y_true[:, 1]) ** 2)
mse_mu = np.mean((y_pred[:, 2] - y_true[:, 2]) ** 2)

print("
📉 Error Metrics:")
print(f"MSE S(t): {mse_s:.4f}")
print(f"MSE X(t): {mse_x:.4f}")
print(f"MSE μ(t): {mse_mu:.4f}")

# === Plot ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(t_test, y_true[:, 0], label="True S")
plt.plot(t_test, y_pred[:, 0], "--", label="Predicted S")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(t_test, y_true[:, 1], label="True X")
plt.plot(t_test, y_pred[:, 1], "--", label="Predicted X")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(t_test, y_true[:, 2], label="True μ")
plt.plot(t_test, y_pred[:, 2], "--", label="Predicted μ")
plt.legend()

plt.tight_layout()
plt.show()


