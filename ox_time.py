from ase.io import read
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# INPUT
# -----------------------------
traj_file = "md.traj"
z_cut = 6.370
tol = 0.01

# -----------------------------
# Load trajectory
# -----------------------------
traj = read(traj_file, ":")

time = []
avg_z_surface = []
count_surface = []

# -----------------------------
# Loop over frames
# -----------------------------
for step, atoms in enumerate(traj):

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    surface_z = []

    for i, sym in enumerate(symbols):

        if sym != "O":
            continue

        z = positions[i][2]

        if z >= z_cut - tol:
            surface_z.append(z)

    # store results
    time.append(step)

    if len(surface_z) > 0:
        avg_z_surface.append(np.mean(surface_z))
    else:
        avg_z_surface.append(0)

    count_surface.append(len(surface_z))

# -----------------------------
# PLOT 1: Average Z vs Time
# -----------------------------
plt.figure()

plt.plot(time, avg_z_surface)

plt.xlabel("Step")
plt.ylabel("Average Z of surface O (Å)")
plt.title("Surface Oxygen Height vs Time")

plt.grid()
plt.savefig("O_surface_z_vs_time.png", dpi=300)

# -----------------------------
# PLOT 2: Count vs Time
# -----------------------------
plt.figure()

plt.plot(time, count_surface)

plt.xlabel("Step")
plt.ylabel("Number of O atoms (z ≥ 6.370 Å)")
plt.title("Surface Oxygen Count vs Time")

plt.grid()
plt.savefig("O_surface_count_vs_time.png", dpi=300)

plt.show()

print("✅ Plots saved!")
