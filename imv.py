import numpy as np
from mendeleev import element
import math
from rdkit import Chem
from rdkit.Chem import AllChem

def readXYZ(filename):

    with open(filename, 'r') as f:

        lines = f.readlines()
        natoms = int(lines[0])
        coords = []
        atoms_symbol = []
        for i in range(2, natoms+2):
            line = lines[i].split()
            coords.append([int(element(line[0]).atomic_weight), float(line[1]), float(line[2]), float(line[3])])
            atoms_symbol.append(line[0])

        coords = np.array(coords)
  
        atoms = coords[:, 0]

        w_coord_x = atoms ** 0.5 * coords[:, 1]
        w_coord_y = atoms ** 0.5 * coords[:, 2]
        w_coord_z = atoms ** 0.5 * coords[:, 3]

        w_coords = np.column_stack((w_coord_x, w_coord_y, w_coord_z))

    return natoms, atoms_symbol, coords, w_coords

def genTransGeoms(filename1, filename2, ntrans):
    # Generate ntrans transition geometries between two geometries
    # filename1: initial geometry
    # filename2: final geometry
    # ntrans: number of transition geometries


    # Read initial and final geometries
    natoms, symbols, init_coords, init_w_coords = readXYZ(filename1)
    natoms, symbols, end_coords, end_w_coords = readXYZ(filename2)

    atoms = init_coords[:, 0]

    assert (init_coords[:, 0] == end_coords[:, 0]).all(), "The two geometries must have the same atoms"

    # Calculate the difference between the initial and final geometries

    diff_w_coords = end_w_coords - init_w_coords

    # Generate ntrans transition geometries

    trans_w_coords = [init_w_coords + i/ntrans * diff_w_coords for i in range(ntrans)]

    trans_coords = [np.column_stack((atoms, w_coord / np.tile(atoms, (3, 1)).T ** 0.5)) for w_coord in trans_w_coords]

    return symbols, trans_coords

def coords2xyzblock(symbols, coords):
    # Convert coordinates to xyz block
    # coords: numpy array of shape (natoms, 3)
    # returns: xyz block as a string
    natoms = coords.shape[0]
    xyzblock = f'{natoms}\n\n'
    for i, coord in enumerate(coords):
        xyzblock += f'{symbols[i]} {coord[1]} {coord[2]} {coord[3]}\n'

    return xyzblock

def genVolumes(filename1, filename2, ntrans):
    symbs, trans_geoms = genTransGeoms(filename1, filename2, ntrans)
    trans_xyzs = [coords2xyzblock(symbs, geom) for geom in trans_geoms]
    mols = [Chem.rdmolfiles.MolFromXYZBlock(xyz) for xyz in trans_xyzs]
    volumes = np.array([AllChem.ComputeMolVolume(mol) for mol in mols])

    return volumes

# init_geom = readXYZ('opt-gs-hex.xyz')
# end_geom = readXYZ('opt-s2-hex.xyz')

init_geom = 'opt-gs-mecn.xyz'
pt_end_geom = 'opt-s2-mecn.xyz'
ct_end_geom = 'opt-s1-mecn.xyz'

pt_volume = genVolumes(init_geom, pt_end_geom, 50)
ct_volume = genVolumes(init_geom, ct_end_geom, 50)

np.savetxt('pt.dat', pt_volume)
np.savetxt('ct.dat', ct_volume)

