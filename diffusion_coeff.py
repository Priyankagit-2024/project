import numpy as np
from ase.io import read

# -----------------------------
# READ TRAJECTORY
# -----------------------------
traj = read("md.traj", "::10")

n_frames = len(traj)
print("Total frames:", n_frames)

# -----------------------------
# TIMESTEP
# -----------------------------
timestep_fs = 1.0 * 10
timestep_ps = timestep_fs * 1e-3

# -----------------------------
# INITIAL STRUCTURE
# -----------------------------
atoms0 = traj[0]
positions = atoms0.get_positions()
symbols = atoms0.get_chemical_symbols()

z = positions[:, 2]

# -----------------------------
# LAYER DETECTION (YOUR WORKING ONE)
# -----------------------------
tolerance = 0.6

layers = []
unused = list(range(len(z)))

while unused:
    ref = unused[0]
    layer = [i for i in unused if abs(z[i] - z[ref]) < tolerance]
    layers.append(layer)
    unused = [i for i in unused if i not in layer]

layers = sorted(layers, key=lambda l: np.mean(z[l]), reverse=True)

print(f"Detected {len(layers)} layers")

for i, layer in enumerate(layers):
    print(f"Layer {i}: {len(layer)} atoms, z_avg = {np.mean(z[layer]):.3f}")

# -----------------------------
# CORRECT LAYER ASSIGNMENT
# -----------------------------
surface_Cu = [i for i in layers[0] if symbols[i] == "Cu"]

subsurface_Cu = [i for i in layers[2] if symbols[i] == "Cu"]   # ✅ FIXED
subsurface_O1 = [i for i in layers[1] if symbols[i] == "O"]
subsurface_O2 = [i for i in layers[3] if symbols[i] == "O"]     # ✅ FIXED

# all Cu
all_Cu = [i for i, s in enumerate(symbols) if s == "Cu"]

print("Surface Cu:", len(surface_Cu))
print("Subsurface Cu:", len(subsurface_Cu))
print("Subsurface O1:", len(subsurface_O1))
print("Subsurface O2:", len(subsurface_O2))

# -----------------------------
# MANUAL DIFFUSION FUNCTION
# -----------------------------
def compute_diffusion(indices, label):

    if len(indices) == 0:
        print(f"{label}: No atoms found")
        return None

    positions = np.array([atoms.get_positions()[indices] for atoms in traj])

    r0 = positions[0]

    msd = []

    for r in positions:
        dr = r - r0
        msd.append(np.mean(np.sum(dr**2, axis=1)))

    msd = np.array(msd)

    time = np.arange(len(msd)) * timestep_ps

    # skip first 10%
    start = int(0.1 * len(time))

    coeff = np.polyfit(time[start:], msd[start:], 1)
    slope = coeff[0]

    # 🔥 USE 2D DIFFUSION (SURFACE SYSTEM)
    D_A2_ps = slope / 4.0
    D_m2_s = D_A2_ps * 1e-8

    print(f"{label}: {D_m2_s:.6e} m^2/s")

    return D_A2_ps, D_m2_s

# -----------------------------
# COMPUTE (UP TO LAYER 3)
# -----------------------------
results = {}

results["Surface_Cu"] = compute_diffusion(surface_Cu, "Surface Cu")
results["Subsurface_Cu"] = compute_diffusion(subsurface_Cu, "Subsurface Cu")
results["Subsurface_O1"] = compute_diffusion(subsurface_O1, "Subsurface O1")
results["Subsurface_O2"] = compute_diffusion(subsurface_O2, "Subsurface O2")
results["All_Cu"] = compute_diffusion(all_Cu, "All Cu")

# -----------------------------
# SAVE OUTPUT
# -----------------------------
with open("layer_diffusion_results.txt", "w") as f:

    f.write("Layer-wise Diffusion Coefficients\n")
    f.write("---------------------------------\n\n")

    for key, val in results.items():

        if val is None:
            continue

        D_A2_ps, D_m2_s = val

        f.write(f"{key}\n")
        f.write(f"D (Å^2/ps) = {D_A2_ps:.6e}\n")
        f.write(f"D (m^2/s)  = {D_m2_s:.6e}\n\n")

print("\n✅ Results saved to layer_diffusion_results.txt")
