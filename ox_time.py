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
count_surface = []

# -----------------------------
# Loop over frames
# -----------------------------
for step, atoms in enumerate(traj):

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    count = 0

    for i, sym in enumerate(symbols):

        if sym != "O":
            continue

        z = positions[i][2]

        if z >= z_cut - tol:
            count += 1

    time.append(step)
    count_surface.append(count)

# -----------------------------
# INITIAL & FINAL VALUES
# -----------------------------
count_initial = count_surface[0]
count_final   = count_surface[-1]

delta_count = count_final - count_initial

# -----------------------------
# PRINT TO SCREEN
# -----------------------------
print("\n====== Surface Oxygen Analysis ======")
print(f"O atoms at i = 0      : {count_initial}")
print(f"O atoms at i = 1 ns   : {count_final}")
print(f"Change (ΔN)           : {delta_count}")

# -----------------------------
# SAVE TO FILE
# -----------------------------
with open("O_surface_change.txt", "w") as f:

    f.write("Surface Oxygen Count Change Analysis\n")
    f.write("====================================\n\n")

    f.write(f"z cutoff = {z_cut} Å\n\n")

    f.write(f"O atoms at i = 0      : {count_initial}\n")
    f.write(f"O atoms at i = 1 ns   : {count_final}\n")
    f.write(f"Change (ΔN)           : {delta_count}\n\n")

    f.write("Full time evolution:\n")
    f.write("Step    O_count\n")

    for t, c in zip(time, count_surface):
        f.write(f"{t:6d}   {c:5d}\n")

print("\n✅ Output saved to O_surface_change.txt")

# -----------------------------
# OPTIONAL: Plot
# -----------------------------
plt.figure()
plt.plot(time, count_surface)

plt.xlabel("Step")
plt.ylabel("Number of O atoms (z ≥ {:.3f} Å)".format(z_cut))
plt.title("Surface Oxygen Count vs Time")

plt.grid()
plt.savefig("O_surface_count_vs_time.png", dpi=300)
plt.show()
