import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# ---------------- PARAMETERS ---------------- #
traj_file = "cu-pos-1.xyz"
inp_file = "o.inp"

dr = 0.05
r_max = 8.0

# ---------------- READ CELL ---------------- #
cell = []
read_cell = False

with open(inp_file) as f:
    for line in f:
        if "&CELL" in line:
            read_cell = True
            continue
        if "&END CELL" in line:
            break
        if read_cell:
            parts = line.split()
            if parts and parts[0] in ["A","B","C"]:
                cell.append([float(parts[1]), float(parts[2]), float(parts[3])])

cell = np.array(cell)

# ---------------- READ TRAJECTORY ---------------- #
traj = read(traj_file, "::50")   # 🔥 less smoothing
traj = traj[-50:]                # 🔥 use equilibrated frames

for atoms in traj:
    atoms.set_cell(cell)
    atoms.set_pbc(True)

# species
symbols = traj[0].get_chemical_symbols()
unique_species = list(set(symbols))

# density
volume = np.linalg.det(cell)
rho = {sp: symbols.count(sp)/volume for sp in unique_species}

# bins
r_bins = np.arange(0, r_max + dr, dr)
r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

# RDF container
g_r = { (i,j): np.zeros(len(r_centers)) for i in unique_species for j in unique_species }

# ---------------- MAIN LOOP ---------------- #
for atoms in traj:

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue

            sp_i = symbols[i]
            sp_j = symbols[j]

            d = atoms.get_distance(i, j, mic=True)

            if d < r_max:
                bin_idx = int(d / dr)
                g_r[(sp_i, sp_j)][bin_idx] += 1

# ---------------- NORMALIZATION ---------------- #
n_frames = len(traj)

for (sp_i, sp_j), counts in g_r.items():

    N_i = symbols.count(sp_i)

    for k, r in enumerate(r_centers):
        shell_vol = 4 * np.pi * r**2 * dr
        norm = rho[sp_j] * shell_vol * N_i * n_frames

        if norm > 0:
            g_r[(sp_i, sp_j)][k] /= norm

# ---------------- SAVE RDF ---------------- #
np.savetxt("rdf_Cu-Cu.dat",
           np.column_stack([r_centers, g_r[("Cu","Cu")]]))

np.savetxt("rdf_Cu-O.dat",
           np.column_stack([r_centers, g_r[("Cu","O")]]))

# ---------------- FIND PEAKS ---------------- #
def find_peaks(r, g):

    peaks = []
    for i in range(1, len(g)-1):
        if g[i] > g[i-1] and g[i] > g[i+1]:
            peaks.append((r[i], g[i]))

    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
    return peaks[:3]   # top 3 peaks

# ---------------- WRITE PEAKS ---------------- #
with open("rdf_peaks.txt", "w") as f:

    for pair in [("Cu","Cu"), ("Cu","O")]:

        peaks = find_peaks(r_centers, g_r[pair])

        f.write(f"Top peaks for {pair[0]}-{pair[1]}:\n")
        for r_p, g_p in peaks:
            f.write(f"  r = {r_p:.3f} Å, g(r) = {g_p:.4f}\n")
        f.write("\n")

# ---------------- PLOT ---------------- #
plt.figure()

plt.plot(r_centers, g_r[("Cu","Cu")], label="Cu-Cu")
plt.plot(r_centers, g_r[("Cu","O")], label="Cu-O")
plt.yticks(fontsize=12, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.xlabel("r (Å)", fontsize=14, fontweight='bold')
plt.ylabel("g(r)", fontsize=14, fontweight='bold')
plt.legend()

plt.savefig("rdf_correct.png", dpi=300)
plt.show()

print("✅ RDF sharper + peaks saved in rdf_peaks.txt")
