import torch
import pandas as pd

# from qml import fchl
from dscribe.descriptors import SOAP, MBTR
from ase import Atoms

from tqdm import tqdm

periodic_table = pd.read_csv('data/periodic_table.csv')

def get_fingerprints(df, rcut=5.0, nmax=4, lmax=1):
    ase_structures = []
    labels = {k: [] for k in df.columns if k != 'structure'}

    species = set()
    for idx, entry in tqdm(df.iterrows(), desc="Building materials fingerprints"):
        struct = entry.structure

        coord_matrix = []
        symbols = []
        atomic_numbers = []
        for site in struct._sites:
            coords = site._frac_coords
            coord_matrix.append(coords)

            symbol = str(list(site._species._data.keys())[0])
            symbols.append(symbol)

            atomic_number = periodic_table.AtomicNumber[periodic_table['Symbol'] == symbol].values[-1]
            atomic_numbers.append(atomic_number)

        # ASE Structure
        ase_structure = Atoms(symbols=symbols, positions=coord_matrix, cell=struct._lattice._matrix) 
        ase_structures.append(ase_structure)
        species.update(ase_structure.get_chemical_symbols())

        for col in df.columns:
            if col != 'structure':
                labels[col].append(entry[col])

    # SOAP
    soap = SOAP(
        species=species,
        average='inner',
        rcut=rcut,
        nmax=nmax,
        lmax=lmax,
        periodic=True,
        sparse=False
    )
    soap_dataset = soap.create(ase_structures, n_jobs=1)

    # MBTR 
    mbtr = MBTR(
        species=species,
        k1 = {
            "geometry": {"function": "atomic_number"},
            "grid": {"min": 1, "max": 3, "sigma": 0.1, "n": 3}
        },
        k2 = {
            "geometry": {"function": "inverse_distance"},
            "grid": {"min": 0.1, "max": 2, "sigma": 0.1, "n": 3},
            "weighting": {"function": "exp", "r_cut": 3, "threshold": 1e-2}
        },  
        k3 = {
            "geometry": {"function": "angle"},
            "grid": {"min": 0, "max": 180, "sigma": 5, "n": 2},
            "weighting" : {"function": "exp", "r_cut": 3, "threshold": 1e-3}
        },
        periodic=True,
        normalization="none"
    )
    mbtr_dataset = mbtr.create(ase_structures)

    return soap_dataset, mbtr_dataset, labels
