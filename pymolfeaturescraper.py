##WORK IN PROGRESS, STILL ACTIVELY EDITING AND ADJUSTING ENVIRONMENT TO TRAIN ENCODER##
from rdkit import Chem
import torch

def smile_convert_to_mol(chemical_formula):
    return Chem.MolFromSmiles(chemical_formula)

def feature_collector(mol,undirected=False,is3D=False):
    node_matrix = []
    edge_vectors = []
    edge_index_matrix = [[],[]]

    for atom in mol.GetAtoms():
        atomvec = []
        atomvec.append(atom.GetAtomicNum())
        atomvec.append(atom.GetFormalCharge())
        atomvec.append(int(atom.GetIsAromatic()))
        atomvec.append(atom.GetTotalNumHs())
        atomvec.append(atom.GetDegree())

        hyb = atom.GetHybridization()
        hybridization_features = [
            int(hyb == Chem.rdchem.HybridizationType.SP),
            int(hyb == Chem.rdchem.HybridizationType.SP2),
            int(hyb == Chem.rdchem.HybridizationType.SP3),
        ]
        atomvec.extend(hybridization_features)

        node_matrix.append(atomvec)

    for bond in mol.GetBonds():

        bondvec = []
        bondvec.append(int(bond.IsInRing()))
        bondvec.append(int(bond.GetIsConjugated()))

        b_type = bond.GetBondType()
        type_features = [
        b_type == Chem.rdchem.BondType.SINGLE,
        b_type == Chem.rdchem.BondType.DOUBLE,
        b_type == Chem.rdchem.BondType.TRIPLE,
        b_type == Chem.rdchem.BondType.AROMATIC
    ]
        bondvec.extend(type_features)

        if is3D:
            b_stereo = bond.GetStereo()
            stereo_features = [
            b_stereo == Chem.rdchem.BondStereo.STEREONONE,
            b_stereo == Chem.rdchem.BondStereo.STEREOANY,
            b_stereo == Chem.rdchem.BondStereo.STEREOZ,
            b_stereo == Chem.rdchem.BondStereo.STEREOE,
        ]
            bondvec.extend(stereo_features)

        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        edge_index_matrix[0].append(start_idx)
        edge_index_matrix[1].append(end_idx)
        edge_vectors.append(bondvec)

        if undirected:
            edge_index_matrix[0].append(end_idx)
            edge_index_matrix[1].append(start_idx)
            edge_vectors.append(bondvec)

    node_matrix = torch.tensor(node_matrix,dtype=torch.float)
    edge_vectors = torch.tensor(edge_vectors,dtype=torch.float)
    edge_index_matrix = torch.tensor(edge_index_matrix,dtype=torch.long)

    return node_matrix, edge_vectors, edge_index_matrix
