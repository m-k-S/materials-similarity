import torch
import pandas as pd

# from qml import fchl
from dscribe.descriptors import SOAP, MBTR
from ase import Atoms

from tqdm import tqdm

periodic_table = pd.read_csv('data/periodic_table.csv')

def get_largest_element(df):
    largest_element = 0
    for idx, entry in df.iterrows():
        struct = entry.structure
        for site in struct._sites:
            symbol = str(list(site._species._data.keys())[0])
            atomic_number = periodic_table.AtomicNumber[periodic_table['Symbol'] == symbol].values[-1]
            if atomic_number > largest_element:
                largest_element = atomic_number
    return largest_element

def get_fingerprints(df, rcut=6.0, nmax=5, lmax=2):
    max_lattice = max([len(i._sites) for i in df.structure])
    max_element = get_largest_element(df)
    data = []

    ase_structures = []
    labels_full = []
    # fchl = []

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

        # FCHL
        # fchl_embedding = fchl.generate_representation(coord_matrix, atomic_numbers, cell=struct._lattice._matrix, max_size=max_lattice, neighbors=100).flatten() 
        # fchl.append(fchl_embedding)

        # ASE Structure
        ase_structure = Atoms(symbols=symbols, positions=coord_matrix, cell=struct._lattice._matrix) 
        ase_structures.append(ase_structure)
        species.update(ase_structure.get_chemical_symbols())

        labels = {}
        for col in df.columns:
            if col != 'structure':
                labels[col] = torch.tensor(entry[col])
        labels_full.append(labels)

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
            "grid": {"min": 1, "max": 10, "sigma": 0.1, "n": 10}
        },
        k2 = {
            "geometry": {"function": "inverse_distance"},
            "grid": {"min": 0.1, "max": 2, "sigma": 0.1, "n": 10},
            "weighting": {"function": "exp", "r_cut": 6, "threshold": 1e-2}
        },  
        k3 = {
            "geometry": {"function": "angle"},
            "grid": {"min": 0, "max": 180, "sigma": 5, "n": 10},
            "weighting" : {"function": "exp", "r_cut": 6, "threshold": 1e-3}
        },
        periodic=True,
        normalization="none"
    )
    mbtr_dataset = mbtr.create(ase_structures)

    return soap_dataset, mbtr_dataset, labels
