#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 04:50:36 2026

@author: marc
"""

"""
Physics-Informed Neural Network (PINN) for the 2D Heat Equation
=================================================================

PDE:            u_t = alpha * (u_xx + u_yy)      for x,y in [0,1], t in [0,1]
Initial cond:   u(x, y, 0) = sin(pi*x) * sin(pi*y)
Boundary cond:  u = 0 on all four edges of the square (x=0, x=1, y=0, y=1)

Analytical solution (used only to check our answer):
    u(x, y, t) = exp(-alpha * 2*pi^2 * t) * sin(pi*x) * sin(pi*y)

This extends the 1D case by simply adding one more spatial dimension.
Everything else about the PINN methodology stays exactly the same:
- the network now takes 3 inputs (x, y, t) instead of 2 (x, t)
- the physics loss now needs u_xx AND u_yy (computed independently)
- we add boundary points on 4 edges instead of 2
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------
# STEP 1: Define constants for the problem
# -----------------------------------------------------------------------
ALPHA = 1.0     # thermal diffusivity coefficient in u_t = alpha * (u_xx + u_yy)


# -----------------------------------------------------------------------
# STEP 2: Define the neural network
# -----------------------------------------------------------------------
# Input:  (x, y, t) -> 3 numbers  (one more than the 1D case)
# Output: u(x, y, t) -> 1 number
# Slightly wider network than the 1D case since the problem is harder
# (more input dimensions to learn a mapping over).
class PINN(nn.Module):
    def __init__(self, layers=[3, 50, 50, 50, 50, 1]):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        return self.net(inputs)


model = PINN().to(device)


# -----------------------------------------------------------------------
# STEP 3: Sample training points
# -----------------------------------------------------------------------
# (a) Collocation points: scattered through the 3D (x, y, t) volume, used
#     to enforce the PDE ("physics loss").
# (b) Initial condition points: t = 0, (x, y) random in the unit square
# (c) Boundary condition points: on the 4 edges of the square, t random

N_COLLOCATION = 20000   # more points than the 1D case since the domain
N_IC = 1000             # is now a 3D volume instead of a 2D area
N_BC = 1000             # (1000 total, spread across the 4 edges)

# (a) Collocation points, sampled randomly inside the (x, y, t) volume
x_f = torch.rand(N_COLLOCATION, 1, device=device, requires_grad=True)
y_f = torch.rand(N_COLLOCATION, 1, device=device, requires_grad=True)
t_f = torch.rand(N_COLLOCATION, 1, device=device, requires_grad=True)

# (b) Initial condition points: t = 0, (x, y) random in [0,1] x [0,1]
x_ic = torch.rand(N_IC, 1, device=device)
y_ic = torch.rand(N_IC, 1, device=device)
t_ic = torch.zeros(N_IC, 1, device=device)
u_ic = torch.sin(np.pi * x_ic) * torch.sin(np.pi * y_ic)   # true IC values

# (c) Boundary condition points: split evenly across the 4 edges
n_per_edge = N_BC // 4

# Edge x = 0
x_b1 = torch.zeros(n_per_edge, 1, device=device)
y_b1 = torch.rand(n_per_edge, 1, device=device)

# Edge x = 1
x_b2 = torch.ones(n_per_edge, 1, device=device)
y_b2 = torch.rand(n_per_edge, 1, device=device)

# Edge y = 0
x_b3 = torch.rand(n_per_edge, 1, device=device)
y_b3 = torch.zeros(n_per_edge, 1, device=device)

# Edge y = 1
x_b4 = torch.rand(n_per_edge, 1, device=device)
y_b4 = torch.ones(n_per_edge, 1, device=device)

x_bc = torch.cat([x_b1, x_b2, x_b3, x_b4], dim=0)
y_bc = torch.cat([y_b1, y_b2, y_b3, y_b4], dim=0)
t_bc = torch.rand(n_per_edge * 4, 1, device=device)
u_bc = torch.zeros(n_per_edge * 4, 1, device=device)   # true boundary values


# -----------------------------------------------------------------------
# STEP 4: Define the physics loss using automatic differentiation
# -----------------------------------------------------------------------
# Same idea as the 1D case, but now we also need u_yy in addition to u_xx.
def physics_residual(model, x, y, t):
    u = model(x, y, t)

    # First derivatives
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_y = torch.autograd.grad(
        u, y, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    # Second derivatives (differentiate the first derivatives again)
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    u_yy = torch.autograd.grad(
        u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True
    )[0]

    # The PDE says this should be exactly zero everywhere
    residual = u_t - ALPHA * (u_xx + u_yy)
    return residual


# -----------------------------------------------------------------------
# STEP 5: Combine all loss terms
# -----------------------------------------------------------------------
mse = nn.MSELoss()

def compute_loss():
    # Physics loss: residual should be 0 at all collocation points
    res = physics_residual(model, x_f, y_f, t_f)
    loss_physics = mse(res, torch.zeros_like(res))

    # Initial condition loss
    u_ic_pred = model(x_ic, y_ic, t_ic)
    loss_ic = mse(u_ic_pred, u_ic)

    # Boundary condition loss (all 4 edges combined)
    u_bc_pred = model(x_bc, y_bc, t_bc)
    loss_bc = mse(u_bc_pred, u_bc)

    total_loss = loss_physics + loss_ic + loss_bc
    return total_loss, loss_physics.item(), loss_ic.item(), loss_bc.item()


# -----------------------------------------------------------------------
# STEP 6: Train the network
# -----------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 8000   # 2D problems generally need more epochs than 1D to converge

print("Training PINN (2D heat equation)...")
for epoch in range(EPOCHS + 1):
    optimizer.zero_grad()
    loss, l_phys, l_ic, l_bc = compute_loss()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Total: {loss.item():.6f} | "
              f"Physics: {l_phys:.6f} | IC: {l_ic:.6f} | BC: {l_bc:.6f}")

print("Training complete.\n")


# -----------------------------------------------------------------------
# STEP 7: Evaluate against the known analytical solution
# -----------------------------------------------------------------------
def analytical_solution(x, y, t):
    return np.exp(-ALPHA * 2 * (np.pi ** 2) * t) * np.sin(np.pi * x) * np.sin(np.pi * y)

# Fix a time slice to visualize (can't plot all of x, y, t at once in 2D plots)
t_snapshot = 0.1

n_test = 100
x_test = np.linspace(0, 1, n_test)
y_test = np.linspace(0, 1, n_test)
X, Y = np.meshgrid(x_test, y_test)

x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
t_flat = torch.full_like(x_flat, t_snapshot)

with torch.no_grad():
    u_pred = model(x_flat, y_flat, t_flat).cpu().numpy().reshape(n_test, n_test)

u_true = analytical_solution(X, Y, t_snapshot)
error = np.abs(u_pred - u_true)
print(f"At t = {t_snapshot}:")
print(f"Max absolute error vs analytical solution: {error.max():.6f}")
print(f"Mean absolute error vs analytical solution: {error.mean():.6f}")


# -----------------------------------------------------------------------
# STEP 8: Plot results at the fixed time snapshot
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].pcolormesh(X, Y, u_pred, shading='auto', cmap='inferno')
axes[0].set_title(f"PINN Prediction (t={t_snapshot})")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(X, Y, u_true, shading='auto', cmap='inferno')
axes[1].set_title(f"Analytical Solution (t={t_snapshot})")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].pcolormesh(X, Y, error, shading='auto', cmap='viridis')
axes[2].set_title("Absolute Error")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("pinn_heat_equation_2D_results.png", dpi=150)
print("Saved plot to pinn_heat_equation_2D_results.png")