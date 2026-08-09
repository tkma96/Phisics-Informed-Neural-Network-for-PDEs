"""
Physics-Informed Neural Network (PINN) for the 1D Heat Equation
=================================================================

PDE:            u_t = alpha * u_xx        for x in [0, 1], t in [0, 1]
Initial cond:   u(x, 0) = sin(pi * x)
Boundary cond:  u(0, t) = u(1, t) = 0

Analytical solution (used only to check our answer):
    u(x, t) = exp(-alpha * pi^2 * t) * sin(pi * x)

Every step is separated and commented so you can see exactly what's happening.
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
ALPHA = 1.0     # thermal diffusivity coefficient in u_t = alpha * u_xx


# -----------------------------------------------------------------------
# STEP 2: Define the neural network
# -----------------------------------------------------------------------
# Input:  (x, t)  -> 2 numbers
# Output: u(x, t) -> 1 number
# A simple feed-forward network with tanh activations (smooth, good for
# taking derivatives) is the standard choice for PINNs.
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
# STEP 3: Sample training points
# -----------------------------------------------------------------------
# (a) Collocation points: scattered throughout the domain, used to enforce
#     the PDE itself (the "physics loss").
# (b) Initial condition points: t = 0, used to enforce u(x,0) = sin(pi*x)
# (c) Boundary condition points: x = 0 and x = 1, used to enforce u = 0

N_COLLOCATION = 10000
N_IC = 200
N_BC = 200

# (a) Collocation points, sampled randomly in the interior of the domain
x_f = torch.rand(N_COLLOCATION, 1, device=device, requires_grad=True)
t_f = torch.rand(N_COLLOCATION, 1, device=device, requires_grad=True)

# (b) Initial condition points: t = 0, x random in [0,1]
x_ic = torch.rand(N_IC, 1, device=device)
t_ic = torch.zeros(N_IC, 1, device=device)
u_ic = torch.sin(np.pi * x_ic)   # true initial condition values

# (c) Boundary condition points: x = 0 or x = 1, t random in [0,1]
t_bc = torch.rand(N_BC, 1, device=device)
x_bc0 = torch.zeros(N_BC, 1, device=device)   # x = 0
x_bc1 = torch.ones(N_BC, 1, device=device)    # x = 1
u_bc = torch.zeros(N_BC, 1, device=device)    # true boundary values (both 0)


# -----------------------------------------------------------------------
# STEP 4: Define the physics loss using automatic differentiation
# -----------------------------------------------------------------------
# This is the heart of a PINN: we compute u_t and u_xx by differentiating
# the network's own output with respect to its inputs, then check how well
# they satisfy u_t - alpha * u_xx = 0.
def physics_residual(model, x, t):
    u = model(x, t)

    # First derivative w.r.t. t
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    # First derivative w.r.t. x
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    # Second derivative w.r.t. x (differentiate u_x again)
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]

    # The PDE says this should be exactly zero everywhere
    residual = u_t - ALPHA * u_xx
    return residual


# -----------------------------------------------------------------------
# STEP 5: Combine all loss terms
# -----------------------------------------------------------------------
mse = nn.MSELoss()

def compute_loss():
    # Physics loss: residual should be 0 at all collocation points
    res = physics_residual(model, x_f, t_f)
    loss_physics = mse(res, torch.zeros_like(res))

    # Initial condition loss
    u_ic_pred = model(x_ic, t_ic)
    loss_ic = mse(u_ic_pred, u_ic)

    # Boundary condition loss (both boundaries combined)
    u_bc0_pred = model(x_bc0, t_bc)
    u_bc1_pred = model(x_bc1, t_bc)
    loss_bc = mse(u_bc0_pred, u_bc) + mse(u_bc1_pred, u_bc)

    # Total weighted loss (weights can be tuned; 1.0 each works fine here)
    total_loss = loss_physics + loss_ic + loss_bc
    return total_loss, loss_physics.item(), loss_ic.item(), loss_bc.item()


# -----------------------------------------------------------------------
# STEP 6: Train the network
# -----------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 5000

print("Training PINN...")
for epoch in range(EPOCHS + 1):
    optimizer.zero_grad()
    loss, l_phys, l_ic, l_bc = compute_loss()  # <- one full pass over all points
    loss.backward()
    optimizer.step()                           # <- one weight update = end of 1 epoch

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Total: {loss.item():.6f} | "
              f"Physics: {l_phys:.6f} | IC: {l_ic:.6f} | BC: {l_bc:.6f}")

print("Training complete.\n")


# -----------------------------------------------------------------------
# STEP 7: Evaluate against the known analytical solution
# -----------------------------------------------------------------------
def analytical_solution(x, t):
    return np.exp(-ALPHA * (np.pi ** 2) * t) * np.sin(np.pi * x)

# Build a grid of test points
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
print(f"Max absolute error vs analytical solution: {error.max():.6f}")
print(f"Mean absolute error vs analytical solution: {error.mean():.6f}")


# -----------------------------------------------------------------------
# STEP 8: Plot results (PINN prediction vs analytical vs error)
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].pcolormesh(X, T, u_pred, shading='auto', cmap='inferno')
axes[0].set_title("PINN Prediction")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(X, T, u_true, shading='auto', cmap='inferno')
axes[1].set_title("Analytical Solution")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].pcolormesh(X, T, error, shading='auto', cmap='viridis')
axes[2].set_title("Absolute Error")
axes[2].set_xlabel("x")
axes[2].set_ylabel("t")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("pinn_heat_equation_results.png", dpi=150)
print("Saved plot to pinn_heat_equation_results.png")
