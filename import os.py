import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# --- CONFIGURATION (The Vinnland Aesthetic) ---
COLOR_BOND = (0.0, 1.0, 0.0)  # Neon Green
COLOR_BG   = (0.0, 0.0, 0.0)  # Void Black
IMG_SIZE   = (1000, 1000)     # High Res

# --- THE MOLECULE QUEUE ---
# 1. 4-Pro-DMT (The heavy shoulder)
# 2. Methamphetamine (The S-isomer, "Ice")
# 3. 3-Hydroxyphenazepam (The "3-OH PHZ" metabolite)
molecule_list = [
    ("4-Pro-DMT", "CCCc1cccc2[nH]cc(CCN(C)C)c12"),
    ("Methamphetamine", "CN[C@@H](C)CC1=CC=CC=C1"),
    ("3-OH-Phenazepam", "OC1C(=O)Nc2ccc(Br)cc2C(=N1)c3ccccc3Cl")
]

def render_vinnland(name, smiles):
    print(f"[*] Processing: {name}...")
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return

    # Make it 3D
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        pass

    # Draw it Black & Green
    drawer = rdMolDraw2D.MolDraw2DCairo(IMG_SIZE[0], IMG_SIZE[1])
    opts = drawer.drawOptions()
    drawer.DrawBackground() # Defaults to black in this mode usually
    
    # Force Green Atoms
    opts.updateAtomPalette({i: COLOR_BOND for i in range(100)})
    opts.bondLineWidth = 4
    opts.atomLabelFontSize = 24
    
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    filename = f"{name.lower().replace(' ', '_')}_vinnland.png"
    with open(filename, "wb") as f:
        f.write(drawer.GetDrawingText())
    print(f"[+] Saved: {filename}")

if __name__ == "__main__":
    for name, smiles in molecule_list:
        render_vinnland(name, smiles)