#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 09:29:12 2026

@author: marc
"""

"""
Inverse-Problem PINN: Discovering an Unknown Diffusivity (alpha)
===================================================================

Forward problem (what we did before): given alpha, find u(x,t).
Inverse problem (this script):        given some measurements of u(x,t),
                                       find the unknown alpha.

Setup:
    True PDE:  u_t = alpha_true * u_xx      (alpha_true = 1.0, but the
                                              network is NOT told this)
    Domain:    x in [0,1], t in [0,1]
    IC:        u(x,0) = sin(pi*x)
    BC:        u(0,t) = u(1,t) = 0

We pretend alpha is unknown. Instead, we generate a handful of noisy
"sensor" measurements of u(x,t) (as if collected from a real experiment),
and let the PINN infer alpha simultaneously while it learns to fit u(x,t).

Key idea: alpha becomes a *trainable parameter* of the model, updated by
the same gradient descent that updates the network's weights. The physics
loss uses the network's CURRENT guess of alpha at every step, so as the
network's solution gets more consistent with both the data and the PDE,
alpha is pulled toward the value that makes both agree.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------
# STEP 1: Ground truth (only used to GENERATE synthetic data, and to
# check our final answer -- the PINN itself never sees this value)
# -----------------------------------------------------------------------
ALPHA_TRUE = 1.0

def analytical_solution(x, t):
    return np.exp(-ALPHA_TRUE * (np.pi ** 2) * t) * np.sin(np.pi * x)


# -----------------------------------------------------------------------
# STEP 2: Generate synthetic "sensor" measurements
# -----------------------------------------------------------------------
# In a real inverse problem, this would be actual experimental data
# (e.g. thermocouple readings). Here we simulate it by sampling the known
# analytical solution and adding a bit of noise, mimicking measurement error.
N_DATA = 100
NOISE_STD = 0.01

x_data_np = np.random.rand(N_DATA, 1)
t_data_np = np.random.rand(N_DATA, 1)
u_data_np = analytical_solution(x_data_np, t_data_np)
u_data_np += np.random.normal(0, NOISE_STD, u_data_np.shape)   # measurement noise

x_data = torch.tensor(x_data_np, dtype=torch.float32, device=device)
t_data = torch.tensor(t_data_np, dtype=torch.float32, device=device)
u_data = torch.tensor(u_data_np, dtype=torch.float32, device=device)


# -----------------------------------------------------------------------
# STEP 3: Define the neural network (same architecture as the 1D forward
# case -- this part of the PINN doesn't change)
# -----------------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, layers=[2, 32, 32, 32, 1]):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


model = PINN().to(device)


# -----------------------------------------------------------------------
# STEP 4: Make alpha a TRAINABLE parameter
# -----------------------------------------------------------------------
# This is the key difference from the forward problem. Instead of a fixed
# Python float, alpha is now a torch.nn.Parameter -- an entry in the
# computation graph that gradient descent can update, just like the
# network's weights.
#
# We start from a deliberately WRONG initial guess (0.5, when the truth
# is 1.0) to demonstrate that the PINN actually learns it from scratch.
#
# We parameterize it as alpha = softplus(alpha_raw) instead of using
# alpha_raw directly. This guarantees alpha stays positive throughout
# training (diffusivity physically cannot be negative), which plain
# unconstrained optimization would not otherwise guarantee.
alpha_raw = nn.Parameter(torch.tensor(0.5, device=device))

def get_alpha():
    return torch.nn.functional.softplus(alpha_raw)


# -----------------------------------------------------------------------
# STEP 5: Sample collocation points for the physics loss
# -----------------------------------------------------------------------
# Same as the forward problem -- points scattered through the domain
# where we enforce the PDE. Note we do NOT need separate IC/BC points
# here: because we have direct data everywhere (including near t=0 and
# the edges by chance), the data loss implicitly constrains those too.
# (You could still add explicit IC/BC terms; we keep it minimal here to
# show the core inverse-problem mechanic clearly.)
N_COLLOCATION = 10000
x_f = torch.rand(N_COLLOCATION, 1, device=device, requires_grad=True)
t_f = torch.rand(N_COLLOCATION, 1, device=device, requires_grad=True)


# -----------------------------------------------------------------------
# STEP 6: Physics residual, now using the CURRENT estimate of alpha
# -----------------------------------------------------------------------
def physics_residual(model, x, t, alpha):
    u = model(x, t)

    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]

    residual = u_t - alpha * u_xx
    return residual


# -----------------------------------------------------------------------
# STEP 7: Combine data loss + physics loss
# -----------------------------------------------------------------------
mse = nn.MSELoss()

def compute_loss():
    alpha = get_alpha()

    # Data loss: how well does the network match the noisy measurements?
    u_pred_data = model(x_data, t_data)
    loss_data = mse(u_pred_data, u_data)

    # Physics loss: does the network's solution satisfy the PDE, using
    # the CURRENT guess for alpha? This is what lets alpha be pulled
    # toward the value consistent with both the data and the physics.
    res = physics_residual(model, x_f, t_f, alpha)
    loss_physics = mse(res, torch.zeros_like(res))

    total_loss = loss_data + loss_physics
    return total_loss, loss_data.item(), loss_physics.item(), alpha.item()


# -----------------------------------------------------------------------
# STEP 8: Train -- note that alpha_raw is included in the optimizer's
# parameter list alongside the network's weights, so it gets updated by
# the exact same gradient descent step.
# -----------------------------------------------------------------------
optimizer = torch.optim.Adam(list(model.parameters()) + [alpha_raw], lr=1e-3)
EPOCHS = 8000

alpha_history = []

print(f"True alpha (hidden from the model): {ALPHA_TRUE}")
print(f"Initial alpha guess:                {get_alpha().item():.4f}\n")
print("Training inverse PINN...")

for epoch in range(EPOCHS + 1):
    optimizer.zero_grad()
    loss, l_data, l_phys, alpha_val = compute_loss()
    loss.backward()
    optimizer.step()

    alpha_history.append(alpha_val)

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Total: {loss.item():.6f} | "
              f"Data: {l_data:.6f} | Physics: {l_phys:.6f} | "
              f"alpha estimate: {alpha_val:.4f}")

print("\nTraining complete.")
print(f"True alpha:      {ALPHA_TRUE:.4f}")
print(f"Recovered alpha: {get_alpha().item():.4f}")
print(f"Relative error:  {abs(get_alpha().item() - ALPHA_TRUE) / ALPHA_TRUE * 100:.2f}%")


# -----------------------------------------------------------------------
# STEP 9: Check the recovered solution field against the true one
# -----------------------------------------------------------------------
n_test = 100
x_test = np.linspace(0, 1, n_test)
t_test = np.linspace(0, 1, n_test)
X, T = np.meshgrid(x_test, t_test)

x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
t_flat = torch.tensor(T.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

with torch.no_grad():
    u_pred = model(x_flat, t_flat).cpu().numpy().reshape(n_test, n_test)

u_true = analytical_solution(X, T)
error = np.abs(u_pred - u_true)
print(f"\nSolution field max absolute error:  {error.max():.6f}")
print(f"Solution field mean absolute error: {error.mean():.6f}")


# -----------------------------------------------------------------------
# STEP 10: Plots -- (1) alpha convergence over training, (2) recovered
# solution field vs ground truth
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Alpha convergence plot
axes[0].plot(alpha_history, label="Estimated alpha", color="tab:blue")
axes[0].axhline(ALPHA_TRUE, color="tab:red", linestyle="--", label="True alpha")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("alpha")
axes[0].set_title("Alpha Convergence")
axes[0].legend()

# Predicted solution field
im1 = axes[1].pcolormesh(X, T, u_pred, shading='auto', cmap='inferno')
axes[1].set_title("Recovered u(x,t)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
fig.colorbar(im1, ax=axes[1])

# Absolute error field
im2 = axes[2].pcolormesh(X, T, error, shading='auto', cmap='viridis')
axes[2].set_title("Absolute Error vs True Solution")
axes[2].set_xlabel("x")
axes[2].set_ylabel("t")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("pinn_inverse_heat_equation_results.png", dpi=150)
print("Saved plot to pinn_inverse_heat_equation_results.png")