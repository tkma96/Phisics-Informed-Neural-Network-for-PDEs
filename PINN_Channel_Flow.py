#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 00:03:44 2026

@author: marc
"""



"""
PINN Surrogate: Impulsively-Started Channel Flow
==================================================

Governing equation (derived in the notes):
    u_t = G + nu * u_yy          for y in [0,1], t in [0, T_MAX]

Initial condition:  u(y, 0) = 0                  (fluid at rest)
Boundary conditions: u(0,t) = u(1,t) = 0          (no-slip walls)

Ground truth (used only for validation, never seen by the network):
    u(y,t) = sum_{n=1,3,5,...} [48/(n^3 pi^3)] sin(n*pi*y) [1 - exp(-n^2 pi^2 * nu * t)]

This script is organized into clearly labeled phases:
    STORAGE SETUP   -> where everything gets saved on your external disk
    TRAINING PHASE  -> only runs if no saved model is found (or if forced)
    INFERENCE PHASE -> loads the trained model and evaluates it on new points
    VISUALIZATION   -> (a) static snapshot figure, (b) animation frames + GIF

Because training is checkpointed to disk, we can re-run this script anytime
and it will skip straight to inference + visualization using the saved model,
without retraining from scratch. Set FORCE_RETRAIN = True below if you want
to retrain anyway (e.g. after changing the network or the physics settings).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# STORAGE SETUP -- everything this script produces is saved under this
# folder on your external disk, so you can safely re-run the script later
# and pick up exactly where you left off.
# =========================================================================
BASE_DIR = "/media/marc/Elements/pinn_channel_flow_data"

MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
FRAMES_DIR = os.path.join(BASE_DIR, "animation_frames")
ANIM_DIR = os.path.join(BASE_DIR, "animation")

for d in [MODEL_DIR, DATA_DIR, PLOTS_DIR, FRAMES_DIR, ANIM_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "pinn_channel_flow.pt")
HISTORY_PATH = os.path.join(DATA_DIR, "training_history.npz")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
EVAL_PATH = os.path.join(DATA_DIR, "eval_results.npz")

# Set this to True if you want to retrain even when a saved model exists
# (e.g. you changed the network architecture, the physics constants, or
# just want a fresh run).
FORCE_RETRAIN = False


# =========================================================================
# PHYSICAL SETUP
# =========================================================================
NU = 1.0                 # viscosity (also sets the diffusive timescale, tau = NU * t)
G = 12.0 * NU             # forcing, chosen so the steady state is exactly 6y(1-y)
T_MAX = 0.5               # covers the flow developing from rest to ~99% of steady state

N_TERMS = 9               # number of odd terms in the analytical series (very accurate)


# =========================================================================
# ANALYTICAL REFERENCE SOLUTION (Eq. 8 from the notes)
# =========================================================================
# Plain NumPy function -- no autograd, no network. This is the ground
# truth we compare the PINN against; it is NEVER used during training.

def analytical_solution(y, t, nu=NU, n_terms=N_TERMS):
    u = np.zeros_like(y, dtype=np.float64)
    for n in range(1, 2 * n_terms, 2):     # n = 1, 3, 5, 7, ...
        coeff = 48.0 / (n ** 3 * np.pi ** 3)
        u += coeff * np.sin(n * np.pi * y) * (1.0 - np.exp(-(n ** 2) * (np.pi ** 2) * nu * t))
    return u


# =========================================================================
# NETWORK DEFINITION
# =========================================================================
# Input:  (y, t) -- two numbers (no x -- the physics doesn't depend on it
#         for this simplified geometry, see notes Section 4)
# Output: u(y, t) -- a single scalar

class PINN(nn.Module):
    def __init__(self, layers=[2, 32, 32, 32, 1]):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, y, t):
        inputs = torch.cat([y, t], dim=1)
        return self.net(inputs)


model = PINN().to(device)


# =========================================================================
# PHYSICS RESIDUAL (used only during training)
# =========================================================================
def physics_residual(model, y, t):
    u = model(y, t)

    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_y = torch.autograd.grad(
        u, y, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_yy = torch.autograd.grad(
        u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True
    )[0]

    residual = u_t - G - NU * u_yy
    return residual


# #########################################################################
# ###########################  TRAINING PHASE  ###########################
# #########################################################################
# Everything in this block only runs if there is no saved model yet (or if
# FORCE_RETRAIN is True). This is the only part of the script that updates
# the network's weights -- everything after this point just uses the
# network as a fixed, trained function.
# #########################################################################

checkpoint_exists = os.path.isfile(MODEL_PATH)

if checkpoint_exists and not FORCE_RETRAIN:
    print(f"Found existing trained model at:\n  {MODEL_PATH}")
    print("Skipping training and loading the saved weights instead.")
    print("(Set FORCE_RETRAIN = True in the script if you want to retrain.)\n")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

else:
    print("No saved model found (or FORCE_RETRAIN is True) -- training from scratch.\n")

    # ---------------------------------------------------------------
    # Sample training points
    # ---------------------------------------------------------------
    N_COLLOCATION = 8000
    N_IC = 500
    N_BC = 500

    # Interior collocation points -- used to enforce the PDE itself
    # Note: we scale t_f by T_MAX BEFORE turning on requires_grad, so it
    # becomes a fresh leaf tensor with no leftover graph history. Doing
    # the scaling on an already-requires_grad tensor instead would leave
    # a stale multiplication node in the graph that breaks on the second
    # training epoch (the previous version of this script had this bug).
    y_f = torch.rand(N_COLLOCATION, 1, device=device, requires_grad=True)
    t_f = (torch.rand(N_COLLOCATION, 1, device=device) * T_MAX).requires_grad_()

    # Initial condition points: t = 0, y random in [0,1]
    y_ic = torch.rand(N_IC, 1, device=device)
    t_ic = torch.zeros(N_IC, 1, device=device)
    u_ic = torch.zeros(N_IC, 1, device=device)      # true IC: u(y,0) = 0

    # Boundary (wall) points: y = 0 or y = 1, t random in [0, T_MAX]
    t_bc = torch.rand(N_BC, 1, device=device) * T_MAX
    y_bc0 = torch.zeros(N_BC, 1, device=device)     # wall at y=0
    y_bc1 = torch.ones(N_BC, 1, device=device)      # wall at y=1
    u_bc = torch.zeros(N_BC, 1, device=device)      # true wall values: u=0

    # ---------------------------------------------------------------
    # Combined loss: physics + initial condition + wall boundary conditions
    # ---------------------------------------------------------------
    mse = nn.MSELoss()

    def compute_loss():
        res = physics_residual(model, y_f, t_f)
        loss_physics = mse(res, torch.zeros_like(res))

        u_ic_pred = model(y_ic, t_ic)
        loss_ic = mse(u_ic_pred, u_ic)

        u_bc0_pred = model(y_bc0, t_bc)
        u_bc1_pred = model(y_bc1, t_bc)
        loss_bc = mse(u_bc0_pred, u_bc) + mse(u_bc1_pred, u_bc)

        total = loss_physics + loss_ic + loss_bc
        return total, loss_physics.item(), loss_ic.item(), loss_bc.item()

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    EPOCHS = 6000

    loss_history = {"epoch": [], "total": [], "physics": [], "ic": [], "bc": []}

    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad()
        loss, l_phys, l_ic, l_bc = compute_loss()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            loss_history["epoch"].append(epoch)
            loss_history["total"].append(loss.item())
            loss_history["physics"].append(l_phys)
            loss_history["ic"].append(l_ic)
            loss_history["bc"].append(l_bc)

        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | Total: {loss.item():.6f} | "
                  f"Physics: {l_phys:.6f} | IC: {l_ic:.6f} | BC: {l_bc:.6f}")

    print("\nTraining complete.\n")

    # ---------------------------------------------------------------
    # Save everything needed to resume/inspect this run later, without
    # retraining: the trained weights, the loss curve, and the physical
    # settings used (so you always know exactly what a saved model
    # corresponds to).
    # ---------------------------------------------------------------
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved trained model weights to:\n  {MODEL_PATH}")

    np.savez(
        HISTORY_PATH,
        epoch=np.array(loss_history["epoch"]),
        total=np.array(loss_history["total"]),
        physics=np.array(loss_history["physics"]),
        ic=np.array(loss_history["ic"]),
        bc=np.array(loss_history["bc"]),
    )
    print(f"Saved training loss history to:\n  {HISTORY_PATH}")

    config = {
        "NU": NU, "G": G, "T_MAX": T_MAX, "N_TERMS": N_TERMS,
        "N_COLLOCATION": N_COLLOCATION, "N_IC": N_IC, "N_BC": N_BC,
        "EPOCHS": EPOCHS, "architecture": [2, 32, 32, 32, 1],
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved run configuration to:\n  {CONFIG_PATH}\n")


# #########################################################################
# ###########################  INFERENCE PHASE  ##########################
# #########################################################################
# From this point on, the network's weights are FIXED -- we only ever call
# model(y, t) to read out predictions. No optimizer, no loss, no backward
# pass. This is the "using the surrogate" phase, as opposed to the
# "building the surrogate" phase above.
# #########################################################################

model.eval()   # tells layers like dropout/batchnorm (not used here, but
               # good practice) that we're in inference mode

# ---------------------------------------------------------------
# Build a fine (y, t) grid and evaluate both the PINN and the
# analytical reference solution on it -- this is the data used for
# every plot and the animation below.
# ---------------------------------------------------------------
N_Y, N_T = 200, 200
y_grid = np.linspace(0, 1, N_Y)
t_grid = np.linspace(0, T_MAX, N_T)
Y, T = np.meshgrid(y_grid, t_grid)   # shape (N_T, N_Y)

y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
t_flat = torch.tensor(T.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

with torch.no_grad():   # no gradients needed for pure inference -- saves memory/time
    u_pinn = model(y_flat, t_flat).cpu().numpy().reshape(N_T, N_Y)

u_ref = analytical_solution(Y, T)
error = np.abs(u_pinn - u_ref)

print(f"Full-field max absolute error:  {error.max():.6f}")
print(f"Full-field mean absolute error: {error.mean():.6f}\n")

# Save the evaluation results too, so re-plotting later doesn't require
# re-running the network at all.
np.savez(
    EVAL_PATH,
    y_grid=y_grid, t_grid=t_grid,
    u_pinn=u_pinn, u_ref=u_ref, error=error,
)
print(f"Saved evaluation grids to:\n  {EVAL_PATH}\n")


# #########################################################################
# ##########################  VISUALIZATION  #############################
# #########################################################################
# Two deliverables, as requested:
#   (a) a static snapshot figure (full-field heatmaps + profile snapshots)
#   (b) an animation: individual frame PNGs (for embedding in a Beamer
#       LaTeX presentation via the `animate` package) plus a combined GIF
#       for quick previewing outside LaTeX.
# Both always show surrogate (PINN), reference (analytical), and error
# together -- never the PINN result on its own.
# #########################################################################

# -------------------------------------------------------------------
# (a) STATIC SNAPSHOT FIGURE
# -------------------------------------------------------------------
fig = plt.figure(figsize=(15, 8))

# Top row: full (y,t) field heatmaps -- PINN / reference / error
ax1 = fig.add_subplot(2, 3, 1)
im1 = ax1.pcolormesh(Y, T, u_pinn, shading="auto", cmap="inferno", vmin=0, vmax=1.6)
ax1.set_title("PINN Surrogate u(y,t)")
ax1.set_xlabel("y"); ax1.set_ylabel("t")
fig.colorbar(im1, ax=ax1)

ax2 = fig.add_subplot(2, 3, 2)
im2 = ax2.pcolormesh(Y, T, u_ref, shading="auto", cmap="inferno", vmin=0, vmax=1.6)
ax2.set_title("Analytical Reference u(y,t)")
ax2.set_xlabel("y"); ax2.set_ylabel("t")
fig.colorbar(im2, ax=ax2)

ax3 = fig.add_subplot(2, 3, 3)
im3 = ax3.pcolormesh(Y, T, error, shading="auto", cmap="viridis")
ax3.set_title("Absolute Error")
ax3.set_xlabel("y"); ax3.set_ylabel("t")
fig.colorbar(im3, ax=ax3)

# Bottom row: velocity profiles u(y) at a few chosen times, PINN vs
# reference overlaid -- this is the clearest way to see the profile
# "developing" from flat/zero into the parabola.
snapshot_times = [0.02, 0.08, 0.2, T_MAX]

ax4 = fig.add_subplot(2, 2, 3)
for st in snapshot_times:
    y_line = torch.tensor(y_grid, dtype=torch.float32, device=device).unsqueeze(1)
    t_line = torch.full_like(y_line, st)
    with torch.no_grad():
        u_pinn_line = model(y_line, t_line).cpu().numpy().flatten()
    u_ref_line = analytical_solution(y_grid, st)

    line, = ax4.plot(y_grid, u_pinn_line, "-", label=f"PINN t={st}")
    ax4.plot(y_grid, u_ref_line, "--", color=line.get_color(), label=f"Ref t={st}")

ax4.set_xlabel("y"); ax4.set_ylabel("u")
ax4.set_title("Profile Snapshots (solid=PINN, dashed=reference)")
ax4.legend(fontsize=7, ncol=2)

# Centerline time-trace: u(y=0.5, t) vs t -- shows the whole transient
# at a glance in a single curve.
ax5 = fig.add_subplot(2, 2, 4)
y_center = torch.full((N_T, 1), 0.5, dtype=torch.float32, device=device)
t_center = torch.tensor(t_grid, dtype=torch.float32, device=device).unsqueeze(1)
with torch.no_grad():
    u_pinn_center = model(y_center, t_center).cpu().numpy().flatten()
u_ref_center = analytical_solution(np.full_like(t_grid, 0.5), t_grid)

ax5.plot(t_grid, u_pinn_center, "-", label="PINN", color="tab:blue")
ax5.plot(t_grid, u_ref_center, "--", label="Reference", color="tab:red")
ax5.set_xlabel("t"); ax5.set_ylabel("u(y=0.5, t)")
ax5.set_title("Centerline Velocity Over Time")
ax5.legend()

# (Using subplots_adjust instead of tight_layout here: this figure mixes a
# 2x3 grid on top with a 2x2 grid on the bottom, and tight_layout warns
# about that grid mismatch even though the figure renders correctly.)
fig.subplots_adjust(hspace=0.4, wspace=0.35)
snapshot_path = os.path.join(PLOTS_DIR, "pinn_channel_flow_snapshots.png")
plt.savefig(snapshot_path, dpi=150)
plt.close(fig)
print(f"Saved static snapshot figure to:\n  {snapshot_path}\n")


# -------------------------------------------------------------------
# (b) ANIMATION -- frame sequence (Beamer-ready) + combined GIF (quick preview)
# -------------------------------------------------------------------
N_FRAMES = 60
frame_times = np.linspace(0, T_MAX, N_FRAMES)

y_line = torch.tensor(y_grid, dtype=torch.float32, device=device).unsqueeze(1)

frame_paths = []
gif_frames = []

for i, ft in enumerate(frame_times):
    t_line = torch.full_like(y_line, float(ft))
    with torch.no_grad():
        u_pinn_line = model(y_line, t_line).cpu().numpy().flatten()
    u_ref_line = analytical_solution(y_grid, ft)
    err_line = np.abs(u_pinn_line - u_ref_line)

    fig_a, (axp, axe) = plt.subplots(1, 2, figsize=(9, 4))

    # Left panel: PINN vs reference profile, fixed axis limits across all
    # frames so the animation doesn't jitter/rescale between frames.
    axp.plot(y_grid, u_pinn_line, "-", color="tab:blue", linewidth=2, label="PINN")
    axp.plot(y_grid, u_ref_line, "--", color="tab:red", linewidth=2, label="Reference")
    axp.set_xlim(0, 1); axp.set_ylim(0, 1.6)
    axp.set_xlabel("y"); axp.set_ylabel("u")
    axp.set_title(f"Velocity Profile,  t = {ft:.3f}")
    axp.legend(loc="upper right")

    # Right panel: absolute error, fixed y-axis scale across frames
    axe.plot(y_grid, err_line, "-", color="black")
    axe.set_xlim(0, 1); axe.set_ylim(0, max(0.05, error.max() * 1.1))
    axe.set_xlabel("y"); axe.set_ylabel("|error|")
    axe.set_title("Absolute Error")

    plt.tight_layout()

    # Naming convention "frame0.png", "frame1.png", ... matches what the
    # LaTeX `animate` package expects for \animategraphics (see usage note
    # printed at the end of this script).
    frame_path = os.path.join(FRAMES_DIR, f"frame{i}.png")
    plt.savefig(frame_path, dpi=120)
    plt.close(fig_a)

    frame_paths.append(frame_path)
    gif_frames.append(Image.open(frame_path).convert("P", palette=Image.ADAPTIVE))

print(f"Saved {N_FRAMES} animation frames to:\n  {FRAMES_DIR}")

# Combine into a single animated GIF too, purely for quick previewing
# outside of LaTeX (e.g. double-click to view, or drop into a chat/email).
gif_path = os.path.join(ANIM_DIR, "pinn_channel_flow.gif")
gif_frames[0].save(
    gif_path,
    save_all=True,
    append_images=gif_frames[1:],
    duration=80,     # milliseconds per frame
    loop=0,          # loop forever
)
print(f"Saved combined preview GIF to:\n  {gif_path}\n")

# -------------------------------------------------------------------
# Usage note for embedding the frame sequence in a Beamer presentation
# -------------------------------------------------------------------

print("=" * 70)
print("To embed the animation in a Beamer/LaTeX presentation:")
print("=" * 70)
print(f"""
1. Copy the whole '{os.path.basename(FRAMES_DIR)}' folder next to your .tex file
   (or reference its full path directly).

2. In your LaTeX preamble:
       \\usepackage{{animate}}

3. In the slide where you want the animation:
       \\animategraphics[width=0.8\\linewidth,autoplay,loop]{{15}}{{{FRAMES_DIR}/frame}}{{0}}{{{N_FRAMES - 1}}}

   - "15" is the playback frame rate (frames per second) -- adjust to taste.
   - "autoplay,loop" starts it automatically and loops it; remove either
     if you'd rather click to start or play once.
   - Must be compiled with pdflatex (not needed to embed a movie player --
     this just flips through the PNG sequence inside the PDF itself, so it
     works in any PDF viewer that supports embedded JavaScript, e.g. Adobe
     Acrobat; some lightweight viewers may only show the first frame).
""")


# -------------------------------------------------------------------
# (c) Second ANIMATION -- a literal picture of "the flow itself": a
# schematic channel (x horizontal, y vertical) colored by velocity
# magnitude, with tracer particles advected downstream at the LOCAL
# velocity u(y,t) -- exactly like dye or seeding particles in a real
# flow-visualization experiment. Faster-moving fluid near the centerline
# visibly outruns the slow fluid near the walls as the parabolic profile
# develops.
#
# 3 panels: PINN surrogate flow / Reference flow / Error color band.
# Because our simplified problem has no x-dependence (u depends only on
# y and t -- see the notes), the channel looks the same at every x at a
# given instant; what makes the animation informative is watching the
# PARTICLES move at different speeds depending on which y-level they're
# on, and watching the PINN's particles gradually line up with (or
# diverge from) the reference's particles as training accuracy allows.
# -------------------------------------------------------------------
CHANNEL_LENGTH = 3.0      # arbitrary visual length of the channel (x-axis)
N_LEVELS = 10             # number of distinct y-levels seeded with tracers
N_PER_LEVEL = 8           # tracers per level (evenly spaced along x)
 
y_levels = np.linspace(0.08, 0.92, N_LEVELS)          # avoid the exact walls
x0 = np.linspace(0, CHANNEL_LENGTH, N_PER_LEVEL, endpoint=False)
 
# Running x-position offset for each y-level's tracers, tracked SEPARATELY
# for the PINN flow and the reference flow, so the two can visibly drift
# apart if the surrogate isn't perfectly accurate yet.
offset_pinn = np.zeros(N_LEVELS)
offset_ref = np.zeros(N_LEVELS)
 
FRAMES_DIR_FLOW = os.path.join(BASE_DIR, "animation_frames_flow")
os.makedirs(FRAMES_DIR_FLOW, exist_ok=True)
 
gif_frames_flow = []
 
# Precompute velocity at each y-level for every frame time (used both to
# advect particles and to color the channel background)
y_levels_t = torch.tensor(y_levels, dtype=torch.float32, device=device).unsqueeze(1)
 
u_max_color = 1.6   # fixed color scale across all frames (0 to 1.6)
 
prev_t = 0.0
for i, ft in enumerate(frame_times):
    dt = ft - prev_t
    prev_t = ft
 
    # Velocity at each tracer y-level, at the current time
    t_col = torch.full_like(y_levels_t, float(ft))
    with torch.no_grad():
        u_levels_pinn = model(y_levels_t, t_col).cpu().numpy().flatten()
    u_levels_ref = analytical_solution(y_levels, ft)
 
    # Advect tracers: x_new = x_old + u * dt (forward Euler), wrapped
    # around the channel length so particles that exit re-enter at x=0
    # (mimics an endlessly-flowing channel, purely for visualization)
    offset_pinn = (offset_pinn + u_levels_pinn * dt) % CHANNEL_LENGTH
    offset_ref = (offset_ref + u_levels_ref * dt) % CHANNEL_LENGTH
 
    # Full velocity profile (fine y_grid) for the background color bands
    y_line_t = torch.tensor(y_grid, dtype=torch.float32, device=device).unsqueeze(1)
    t_line_full = torch.full_like(y_line_t, float(ft))
    with torch.no_grad():
        u_full_pinn = model(y_line_t, t_line_full).cpu().numpy().flatten()
    u_full_ref = analytical_solution(y_grid, ft)
    err_full = np.abs(u_full_pinn - u_full_ref)
 
    fig_c, (axP, axR, axE) = plt.subplots(1, 3, figsize=(14, 4.5))
 
    # --- Panel 1: PINN surrogate flow ---
    color_band_pinn = np.tile(u_full_pinn.reshape(-1, 1), (1, 2))  # constant along x
    axP.imshow(color_band_pinn, extent=[0, CHANNEL_LENGTH, 0, 1], origin="lower",
               aspect="auto", cmap="inferno", vmin=0, vmax=u_max_color)
    for lvl_idx, y_lvl in enumerate(y_levels):
        x_positions = (x0 + offset_pinn[lvl_idx]) % CHANNEL_LENGTH
        axP.scatter(x_positions, np.full_like(x_positions, y_lvl),
                    color="cyan", edgecolor="black", s=35, zorder=5)
    axP.set_title("PINN Surrogate Flow")
    axP.set_xlabel("x (channel length)"); axP.set_ylabel("y")
    axP.set_ylim(0, 1); axP.set_xlim(0, CHANNEL_LENGTH)
 
    # --- Panel 2: Reference flow (same layout, same color scale) ---
    color_band_ref = np.tile(u_full_ref.reshape(-1, 1), (1, 2))
    axR.imshow(color_band_ref, extent=[0, CHANNEL_LENGTH, 0, 1], origin="lower",
               aspect="auto", cmap="inferno", vmin=0, vmax=u_max_color)
    for lvl_idx, y_lvl in enumerate(y_levels):
        x_positions = (x0 + offset_ref[lvl_idx]) % CHANNEL_LENGTH
        axR.scatter(x_positions, np.full_like(x_positions, y_lvl),
                    color="lime", edgecolor="black", s=35, zorder=5)
    axR.set_title("Reference Flow")
    axR.set_xlabel("x (channel length)"); axR.set_ylabel("y")
    axR.set_ylim(0, 1); axR.set_xlim(0, CHANNEL_LENGTH)
 
    # --- Panel 3: error color band (magnitude only, no particles -- error
    # isn't a physical flow, just a mismatch magnitude) ---
    color_band_err = np.tile(err_full.reshape(-1, 1), (1, 2))
    imE = axE.imshow(color_band_err, extent=[0, CHANNEL_LENGTH, 0, 1], origin="lower",
                      aspect="auto", cmap="viridis", vmin=0, vmax=max(0.05, error.max()))
    axE.set_title("Absolute Error")
    axE.set_xlabel("x (channel length)"); axE.set_ylabel("y")
    axE.set_ylim(0, 1); axE.set_xlim(0, CHANNEL_LENGTH)
    fig_c.colorbar(imE, ax=axE, fraction=0.046, pad=0.04)
 
    fig_c.suptitle(f"t = {ft:.3f}   (particle color: cyan = PINN, lime = reference)")
    fig_c.subplots_adjust(wspace=0.4, top=0.85)
 
    frame_path_flow = os.path.join(FRAMES_DIR_FLOW, f"frame{i}.png")
    plt.savefig(frame_path_flow, dpi=120)
    plt.close(fig_c)
 
    gif_frames_flow.append(Image.open(frame_path_flow).convert("P", palette=Image.ADAPTIVE))
 
print(f"Saved {N_FRAMES} flow-visualization animation frames to:\n  {FRAMES_DIR_FLOW}")
 
gif_path_flow = os.path.join(ANIM_DIR, "pinn_channel_flow_particles.gif")
gif_frames_flow[0].save(
    gif_path_flow,
    save_all=True,
    append_images=gif_frames_flow[1:],
    duration=80,
    loop=0,
)
print(f"Saved combined flow-visualization preview GIF to:\n  {gif_path_flow}\n")
 
print("=" * 70)
print("To embed the FLOW-VISUALIZATION animation in a Beamer/LaTeX presentation:")
print("=" * 70)
print(f"""
       \\animategraphics[width=\\linewidth,autoplay,loop]{{15}}{{{FRAMES_DIR_FLOW}/frame}}{{0}}{{{N_FRAMES - 1}}}
""")
 
