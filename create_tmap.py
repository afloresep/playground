import pandas as pd
import tmap
from faerun import Faerun
from mhfp.encoder import MHFPEncoder
from rdkit.Chem import AllChem


df = pd.read_csv('HMDB-smiles-short.csv')
print(df.shape)

# The number of permutations used by the MinHashing algorithm
perm = 512

# Initializing the MHFP encoder with 512 permutations
enc = MHFPEncoder(perm)

# Create MHFP fingerprints from SMILES
# The fingerprint vectors have to be of the tm.VectorUint data type
fingerprints = [tmap.VectorUint(enc.encode(s)) for s in df["smiles"]]

# Initialize the LSH Forest
lf = tmap.LSHForest(perm)

# Add the Fingerprints to the LSH Forest and index
lf.batch_add(fingerprints)
lf.index()

# Get the coordinates
x, y, s, t, _ = tmap.layout_from_lsh_forest(lf)



# Now plot the data
faerun = Faerun(view="front", coords=False)
faerun.add_scatter(
    "ESOL_Basic",
    {   "x": x, 
        "y": y, 
        "c": list(df.logSolubility.values), 
        "labels": df["smiles"]},
    point_scale=5,
    colormap = ['rainbow'],
    has_legend=True,
    legend_title = ['ESOL (mol/L)'],
    categorical=[False],
    shader = 'smoothCircle'
)

faerun.add_tree("ESOL_Basic_tree", {"from": s, "to": t}, point_helper="ESOL_Basic")

# Choose the "smiles" template to display structure on hover
faerun.plot('ESOL_Basic', template="smiles", notebook_height=750)

cs = {'x': x, 'y':y}
ps = {'s':s, 't':t}
pa = pd.DataFrame(cs)
ps = pd.DataFrame(ps)

pa.to_csv('values.csv')
ps.to_csv('values2.csv')