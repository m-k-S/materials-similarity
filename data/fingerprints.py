from qml import fchl
from dscribe.descriptors import SOAP, MBTR
from ase.build import molecule
from ase import Atoms
from tqdm import tqdm

periodic_table = pd.read_csv('data/periodic_table.csv')

def get_features_and_coords(df):
    largest_element = get_largest_element(df)
    data = []

    for idx, entry in tqdm(df.iterrows(), desc="Building material graphs"):
        struct = entry.structure

        feature_matrix = []
        coord_matrix = []

        # Features
        for site in struct._sites:
            feature_vec = [0 for _ in range(largest_element)] # create a vector of zeros
            symbol = str(list(site._species._data.keys())[0])
            atomic_number = periodic_table.AtomicNumber[periodic_table['Symbol'] == symbol].values[-1]
            feature_vec[atomic_number - 1] = 1 # one-hot encode atomic number
            feature_matrix.append(feature_vec)

        # Coordinates
        for site in struct._sites:
            coords = site._frac_coords
            coord_matrix.append(coords)

        coord_matrix = torch.FloatTensor(np.array(coord_matrix))
        feature_matrix = torch.FloatTensor(np.array(feature_matrix))

        # Labels
        labels = {}
        for col in df.columns:
            if col != 'structure':
                labels[col] = torch.tensor(entry[col])

        if (feature_matrix is not None) and (len(feature_matrix) > 1): 
            edge_index=make_edge_indices(entry)
            if len(edge_index.shape) > 1:
                datum = Data(x=feature_matrix, edge_index=edge_index, y=labels, pos=coord_matrix)
                data.append(datum)

def get_fingerprints(df):
    max_lattice = 0
    for d in df:
        if len(d.x) > max_lattice:
            max_lattice = len(d.x)

    for idx, entry in tqdm(df.iterrows(), desc="Building materials fingerprints"):
        struct = entry.structure

        coord_matrix = []
        atomic_numbers = []
        for site in struct._sites:
            coords = site._frac_coords
            coord_matrix.append(coords)

            symbol = str(list(site._species._data.keys())[0])
            atomic_number = periodic_table.AtomicNumber[periodic_table['Symbol'] == symbol].values[-1]
            atomic_numbers.append(atomic_number)

        # FCHL
        fc = fchl.generate_representation(coord_matrix, atomic_numbers, cell=struct._lattice._matrix, max_size=max_num_atoms, neighbors=100).flatten() 

        # SOAP
        structure = Atoms(symbols=symbols, positions=coord_matrix, cell=struct._lattice._matrix) 
