import json
import numpy as np
#from pymatgen import Lattice, Structure, Molecule

def atom_info(at):
    """ Returns atomic number, period and group of each atom."""
    
    # List of all the elements in the periodic table
    list_of_atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
    'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
    'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
    'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
    'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
    'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh',
    'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    
    # Corresponding period of each atom
    period = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    
    # Corresponding group of each atom
    group = [1, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 13, 14, 15, 16, 17, 18,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 4,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    # Get index of the placed at 
    ind = list_of_atoms.index(at)
    
    return ind+1, period[ind], group[ind]


def cell_parameters(df): 
    """Extracts lattice constants a, b, and c (in A) and angles (in radians)
    from one entry of the database."""
    abc = np.array([df["a"], df["b"], df["c"]])
    angles = np.array([df["alpha"], df["beta"], df["gamma"]]) * np.pi/180
    return abc, angles


def unitcell(atoms_list):
    """Returns a list of the form ['Atom',[coordinates]], where coordinates
    are given in the lattice constants."""
    species = [atoms[0] for atoms in atoms_list]
    sorts = list(dict.fromkeys(species))
    
    clA = []
    clB = []
    for atom in atoms_list:
        if atom[0] == sorts[0]:
            clA.append(atom[1])
        else:
            clB.append(atom[1])        
    
    return np.asarray(clA), np.asarray(clB)

def supercell(atoms_list):
    """Creates a supercell comprising 27 copies of the unit cell."""
    species = [atoms[0] for atoms in atoms_list]
    sorts = list(dict.fromkeys(species))
    
    spclA = []
    spclB = []

    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                for atom in atoms_list:
                    if atom[0] == sorts[0]:
                        spclA.append(np.asarray(atom[1]) + np.array([i,j,k]))
                    else:
                        spclB.append(np.asarray(atom[1]) + np.array([i,j,k]))     
    return np.asarray(spclA), np.asarray(spclB)

def neighbours(df, atoms_list):
    """Generates a list of distances to neighbours (NNs) for each pairs of species: AA, AB, BA, BB."""
    
    # The initial unit cell and supercell, separated into sublattices of A and B atoms
    clA, clB = unitcell(atoms_list)
    spclA, spclB = supercell(atoms_list)
    
    # Unit vectors used to reshape the arrays 
    eA = np.ones(len(clA))
    eB = np.ones(len(clB))
    EA = np.ones(len(spclA))
    EB = np.ones(len(spclB))
    
    # Yields the matrix to calcualte the interatomic distances in the lattice coordinate system
    abc = cell_parameters(df)[0]
    ang = np.cos(cell_parameters(df)[1])
    mat = np.array([[1,ang[2],ang[1]],[ang[2],1,ang[0]],[ang[1],ang[0],1]])
    mat = np.multiply(mat,np.tensordot(abc,abc,axes=0))
    
    # Pairwise distances between atoms in the initial cell and supercell    
    drAA = np.tensordot(eA,spclA,axes=0) - np.swapaxes(np.tensordot(EA,clA,axes=0),0,1)
    drAB = np.tensordot(eA,spclB,axes=0) - np.swapaxes(np.tensordot(EB,clA,axes=0),0,1)
    drBA = np.tensordot(eB,spclA,axes=0) - np.swapaxes(np.tensordot(EA,clB,axes=0),0,1)
    drBB = np.tensordot(eB,spclB,axes=0) - np.swapaxes(np.tensordot(EB,clB,axes=0),0,1)
    
    # This part creates a 3x3 matrix for each pair to calculate the distances 
    drAA = np.tensordot(drAA,np.ones(3),axes=0)
    drAB = np.tensordot(drAB,np.ones(3),axes=0)
    drBA = np.tensordot(drBA,np.ones(3),axes=0)
    drBB = np.tensordot(drBB,np.ones(3),axes=0)
    
    drAA = np.multiply(drAA,np.swapaxes(drAA,2,3))
    drAB = np.multiply(drAB,np.swapaxes(drAB,2,3))
    drBA = np.multiply(drBA,np.swapaxes(drBA,2,3))
    drBB = np.multiply(drBB,np.swapaxes(drBB,2,3))
    
    # Calculate interatomic distances using the previously derived matrix     
    distAA = np.multiply(np.tensordot(eA,np.tensordot(EA,mat,axes=0),axes=0),drAA)
    distAB = np.multiply(np.tensordot(eA,np.tensordot(EB,mat,axes=0),axes=0),drAB)
    distBA = np.multiply(np.tensordot(eB,np.tensordot(EA,mat,axes=0),axes=0),drBA)
    distBB = np.multiply(np.tensordot(eB,np.tensordot(EB,mat,axes=0),axes=0),drBB)
    
    distAA = np.sqrt(np.sum(np.sum(distAA,axis=3),axis=2))
    distAB = np.sqrt(np.sum(np.sum(distAB,axis=3),axis=2))
    distBA = np.sqrt(np.sum(np.sum(distBA,axis=3),axis=2))
    distBB = np.sqrt(np.sum(np.sum(distBB,axis=3),axis=2))
    
    return distAA, distAB, distBA, distBB
    
def asymmetric_descriptor(df, atoms_list):
    """A crystal-structure descriptor producing raw values."""
    
    # List of species of atoms A and B in a given binary material
    species = [atoms[0] for atoms in atoms_list]
    sorts = list(dict.fromkeys(species))
    
    # The period P and group G of atoms A and B
    PA, GA = atom_info(sorts[0])[1], atom_info(sorts[0])[2]
    PB, GB = atom_info(sorts[1])[1], atom_info(sorts[1])[2]
    
    # Pairwise distances between atoms in each sublattice
    distAA, distAB, distBA, distBB = neighbours(df, atoms_list)
    
    # The distance to NNs for each distinct atom in the unit cell (distance of an atom to itself)
    rAA = np.partition(distAA,1,axis=1)[:,1]
    rAB = np.partition(distAB,1,axis=1)[:,1]
    rBA = np.partition(distBA,1,axis=1)[:,1]
    rBB = np.partition(distBB,1,axis=1)[:,1]
    
    # Averaged distances to NNs for atoms of the same sort in the unit cell    
    RAA = np.sum(rAA)/len(rAA)
    RAB = np.sum(rAB)/len(rAB)
    RBA = np.sum(rBA)/len(rBA)
    RBB = np.sum(rBB)/len(rBB)
    
    # For counting NNs, i.e., the coordination number
    rAA = np.tensordot(rAA,np.ones(distAA.shape[1]),axes=0)
    rAB = np.tensordot(rAB,np.ones(distAB.shape[1]),axes=0)
    rBA = np.tensordot(rBA,np.ones(distBA.shape[1]),axes=0)
    rBB = np.tensordot(rBB,np.ones(distBB.shape[1]),axes=0)
    
    # A mask choosing neighbours which fall within a layer of 10% of the minimal distance 
    maskAA = (distAA < 1.1*rAA) & (distAA > 0.)
    maskAB = (distAB < 1.1*rAB) & (distAB > 0.)
    maskBA = (distBA < 1.1*rBA) & (distBA > 0.)
    maskBB = (distBB < 1.1*rBB) & (distBB > 0.)
    
    # Number of symmetrically distinct atoms of sorts A and B in the unit cell
    nA = distAA.shape[0]
    nB = distBB.shape[0]
    
    # The coordination numbers averaged over the atoms in the unit cell
    CAA = np.count_nonzero(maskAA)/nA
    CAB = np.count_nonzero(maskAB)/nA
    CBA = np.count_nonzero(maskBA)/nB
    CBB = np.count_nonzero(maskBB)/nB
    
    # Finally return period and group of atoms A and B, their occurernces in the unit cell,
    # coordination number and distances between each sublattices 
    
    return np.array([PA,PB,GA,GB,nA,nB,CAA,CBB,CAB,CBA,RAA,RBB,RAB,RBA])

def descriptor(df, atoms_list):
    """A symmetrised descriptor which is independent on the order of A and B atoms."""
    
    # Load the raw descriptor and reshape it into two vectors xA and xB belonging to A and B atoms
    asym = asymmetric_descriptor(df, atoms_list)    
    asym = asym.reshape((int(len(asym)/2)),2)
    
    # This part transforms periods P and groups G of each element into their combinations (P+G) and (P-G) 
    mask = np.eye(len(asym))
    mask[0,1] = 1
    mask[1,0] = 1
    mask[1,1] = -1
    asym = np.dot(mask,asym)
    
    # Symmetrise the two vectors xA and xB as (xA + xB)/2 and (xA - xB)/2*sign(A+ - B+)   
    x = (asym[:,0]+asym[:,1])/2
    y = (asym[:,0]-asym[:,1])/2
    y = y*np.sign(y[0])
    
    # Merge the two vectors into a single 14-element array
    sym = np.array([x,y])
    sym = np.swapaxes(sym,0,1)
    sym = sym.reshape(len(sym)*2)
    
    return sym