##WORK IN PROGRESS, STILL ACTIVELY EDITING AND ADJUSTING ENVIRONMENT AND HELPER FILES TO TRAIN ENCODER##
import torch
from torch.utils.data import Dataset, DataLoader
# from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from torch_geometric.data import Data
from torch import nn
import esm
import torch.optim as optim
import random

from dataset_cleaner_chembl2 import clean_dataset
from dataset_cleaner_KLIF import datacleaner_KLIF
from pymolfeaturescraper import feature_collector
from evaluate_encodings import evaluate_binding, evaluate_mechanism

from chembl_webresource_client.new_client import new_client

def get_chembl_kinase_targets(organism="Homo sapiens"):
    target = new_client.target

   
    kinase_targets = target.search("kinase").filter(
        target_type="SINGLE PROTEIN",
        target_organism=organism
    ).only(["target_chembl_id", "pref_name"])

    target_ids = sorted({t["target_chembl_id"] for t in kinase_targets if "target_chembl_id" in t})

    print(f"Found {len(target_ids)} human single-protein kinase targets in ChEMBL")
    return target_ids

class ChemblBindDataset(Dataset):
    def __init__(self, df):
        self.smiles_list = df["smiles"].tolist()
        self.seq_list = df["sequence"].tolist()
        self.labels = df["label"].tolist()
    def __len__(self):
        return len(self.smiles_list)
    def __getitem__(self,idx):
        smiles = self.smiles_list[idx]
        seq = self.seq_list[idx]
        label = self.labels[idx]

        node_matrix, edge_vectors, edge_index_matrix = feature_collector(Chem.MolFromSmiles(smiles))
        graph = Data(x=node_matrix,edge_index=edge_index_matrix,edge_attr=edge_vectors)
        return graph, seq, label

class MechBindDataset(Dataset):
    def __init__(self,df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx, retry_depth=0):
        max_retries = 20
        anchor_row = self.df.iloc[idx]
        anchor_smiles = anchor_row['ligand.smiles']
        anchor_seq = anchor_row['kinase.full_sequence']
        anchor_target = anchor_row['kinase.name']

        positive_pool = self.df[(self.df["kinase.name"] == anchor_target)
                        & (self.df["binding_mode_label"] == anchor_row["binding_mode_label"])
                        & (self.df.index != idx)]
        negative_pool = self.df[(self.df["kinase.name"] == anchor_target)
                                & (self.df["binding_mode_label"] != anchor_row["binding_mode_label"])]

        if len(positive_pool) == 0 or len(negative_pool) == 0:
            if retry_depth >= max_retries:
                raise RuntimeError(f"can't find anchor with valid pos and neg partner after {max_retries} attempts, probably too sparse for triplet loss")
            new_indx = random.randrange(len(self.df))
            return self.__getitem__(new_indx,retry_depth=retry_depth+1)

        anchor_node_matrix, anchor_edge_vectors, anchor_edge_index_matrix = feature_collector(Chem.MolFromSmiles(anchor_smiles))
        anchor_graph = Data(x=anchor_node_matrix,edge_index=anchor_edge_index_matrix,edge_attr=anchor_edge_vectors)

        pos_row = positive_pool.sample(1).iloc[0]
        pos_smiles = pos_row['ligand.smiles']
        pos_seq = pos_row['kinase.full_sequence']
        # pos_target = pos_row['kinase.name']

        pos_node_matrix, pos_edge_vectors, pos_edge_index_matrix = feature_collector(Chem.MolFromSmiles(pos_smiles))
        pos_graph = Data(x=pos_node_matrix,edge_index=pos_edge_index_matrix,edge_attr=pos_edge_vectors)

        neg_row = negative_pool.sample(1).iloc[0]
        neg_smiles = neg_row['ligand.smiles']
        neg_seq = neg_row['kinase.full_sequence']
        # neg_target = neg_row['kinase.name']

        neg_node_matrix, neg_edge_vectors, neg_edge_index_matrix = feature_collector(Chem.MolFromSmiles(neg_smiles))
        neg_graph = Data(x=neg_node_matrix,edge_index=neg_edge_index_matrix,edge_attr=neg_edge_vectors)

        return anchor_graph,anchor_seq,pos_graph,pos_seq,neg_graph,neg_seq

_esm_model = None
_esm_alphabet = None
_esm_batch_converter = None

def get_esm_model():
    global _esm_model, _esm_alphabet, _esm_batch_converter
    if _esm_model is None:
        _esm_model, _esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        _esm_batch_converter = _esm_alphabet.get_batch_converter()
    return _esm_model, _esm_alphabet, _esm_batch_converter

def bind_collate_fn(batch):
    graphs,seqs,labels = zip(*batch)
    from torch_geometric.data import Batch
    graph_batch = Batch.from_data_list(graphs)
    data = [(str(i), seq) for i, seq in enumerate(seqs)]
    _, _, batch_converter = get_esm_model()
    _, _, tokens = batch_converter(data)
    labels = torch.tensor(labels,dtype=torch.float)
    return graph_batch,tokens,labels

def mech_collate_fn(batch):
    a_g, a_s, p_g, p_s, n_g, n_s = zip(*batch)
    from torch_geometric.data import Batch
    a_batch = Batch.from_data_list(a_g)
    p_batch = Batch.from_data_list(p_g)
    n_batch = Batch.from_data_list(n_g)
    _, _, batch_converter = get_esm_model()
    a_tok = batch_converter([(str(i), s) for i, s in enumerate(a_s)])[2]
    p_tok = batch_converter([(str(i), s) for i, s in enumerate(p_s)])[2]
    n_tok = batch_converter([(str(i), s) for i, s in enumerate(n_s)])[2]
    return a_batch, a_tok, p_batch, p_tok, n_batch, n_tok

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GINEConv
import torch.nn.functional as F

def mlp(in_channels,out_channels):
    return nn.Sequential(nn.Linear(in_channels,out_channels),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(out_channels,out_channels))

class GNN(nn.Module):
    def __init__(self,input_node_dim,input_edge_dim,hidden_dim):
        super().__init__()

        self.edge_project_input = nn.Linear(input_edge_dim, input_node_dim)
        self.edge_project_hidden = nn.Linear(input_edge_dim, hidden_dim)
        self.conv1 = GINEConv(mlp(input_node_dim,hidden_dim))
        self.conv2 = GINEConv(mlp(hidden_dim,hidden_dim))
        self.conv3 = GINEConv(mlp(hidden_dim,hidden_dim))

    def forward(self,x,edge_index,edge_attr,batch):
        x = x.float()
        edge_attr = edge_attr.float()

        edge_attr_1 = self.edge_project_input(edge_attr)
        x = F.relu(self.conv1(x, edge_index, edge_attr_1))

        edge_attr_hidden = self.edge_project_hidden(edge_attr)
        x = F.relu(self.conv2(x, edge_index, edge_attr_hidden))
        x = F.relu(self.conv3(x, edge_index, edge_attr_hidden))

        x = global_mean_pool(x, batch)
        return x

class LigandEncoder(nn.Module):
    def __init__(self,gnn,hidden_dim):
        super().__init__()
        self.gnn = gnn
        self.project = nn.Linear(hidden_dim,256)
    def forward(self,graph):
        x,edge_index,edge_attr = graph.x, graph.edge_index,graph.edge_attr
        h = self.gnn(x,edge_index,edge_attr,graph.batch)
        z = self.project(h)
        return z

class ProteinEncoder(nn.Module):
    def __init__(self,esm_model,layer=6):
        super().__init__()
        self.esm = esm_model
        self.layer = layer
        self.project = nn.Linear(esm_model.embed_dim,256)

    def forward(self,tokens):
        out = self.esm(tokens, repr_layers=[self.layer])
        h = out["representations"][self.layer]
        h = h[:, 1:-1].mean(dim=1)
        z = self.project(h)
        return z

def similarity(vec1,vec2,temperature):
    vec1_norm = F.normalize(vec1, p=2, dim=-1)
    vec2_norm = F.normalize(vec2, p=2, dim=-1)
    cos_sim = (vec1_norm * vec2_norm).sum(dim=-1)
    return cos_sim / temperature

def bind_encoder_train(bind_loader,val_bind_loader, lig_encoder, prot_encoder, num_epochs,optimizer,criterion,device):
    lig_encoder.train()
    prot_encoder.train()

    for epoch in range(num_epochs):
        for graph, tok, labels in bind_loader:
            graph = graph.to(device)
            tok = tok.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            z_L = lig_encoder(graph)
            z_P = prot_encoder(tok)
            scores = similarity(z_L,z_P,0.05)
            loss = criterion(scores,labels)
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            evaluate_binding(val_bind_loader,lig_encoder,prot_encoder,0.05,device)

def mech_encoder_train(mech_loader,val_mech_loader, lig_encoder, prot_encoder, num_epochs,optimizer,criterion,device):

    lig_encoder.train()
    for epoch in range(num_epochs):
        for (a_graph,a_tok,p_graph,p_tok,n_graph,n_tok) in mech_loader:
            a_graph = a_graph.to(device)
            p_graph = p_graph.to(device)
            n_graph = n_graph.to(device)

            optimizer.zero_grad()
            zA_L = lig_encoder(a_graph)
            zP_L = lig_encoder(p_graph)
            zN_L = lig_encoder(n_graph)

            loss = criterion(zA_L,zP_L,zN_L)

            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            evaluate_mechanism(val_mech_loader,lig_encoder,prot_encoder,device)

from evaluate_encodings import evaluate_binding
from evaluate_encodings import evaluate_mechanism
from dataset_cleaner_chembl2 import split_bind_dataset
from dataset_cleaner_KLIF import split_mech_dataset

def main():
    kinase_targets = get_chembl_kinase_targets()
    chem_data_df = clean_dataset(target_chembl_ids=kinase_targets)
    mech_df = datacleaner_KLIF()

    train_chem_df,val_chem_df,test_chem_df = split_bind_dataset(chem_data_df)
    train_mech_df, val_mech_df,test_mech_df = split_mech_dataset(mech_df)

    train_bind_loader = DataLoader(ChemblBindDataset(train_chem_df),batch_size=64,collate_fn=bind_collate_fn,shuffle=True)
    val_bind_loader = DataLoader(ChemblBindDataset(val_chem_df),batch_size=64,collate_fn=bind_collate_fn,shuffle=True)
    test_bind_loader = DataLoader(ChemblBindDataset(test_chem_df),batch_size=64,collate_fn=bind_collate_fn,shuffle=True)

    train_mech_loader = DataLoader(MechBindDataset(train_mech_df),batch_size=1,collate_fn=mech_collate_fn,shuffle=True)
    val_mech_loader = DataLoader(MechBindDataset(val_mech_df),batch_size=1,collate_fn=mech_collate_fn,shuffle=True)
    test_mech_loader= DataLoader(MechBindDataset(test_mech_df),batch_size=1,collate_fn=mech_collate_fn,shuffle=True)

    input_node_dim = 8
    input_edge_dim = 6
    hidden_dim = 128

    gnn_inst = GNN(input_node_dim,input_edge_dim, hidden_dim)
    ligand_encoder = LigandEncoder(gnn_inst,hidden_dim)
    esm_model, _, _ = get_esm_model()
    protein_encoder = ProteinEncoder(esm_model,layer=6)

    criterion_bind = nn.BCEWithLogitsLoss()
    optimizer_bind = optim.AdamW(
        list(ligand_encoder.parameters()) + list(protein_encoder.parameters()),
        lr=1e-3,
        weight_decay=1e-2)

    epchs = 100
    dev = 'cpu'

    bind_encoder_train(train_bind_loader,val_bind_loader,ligand_encoder,protein_encoder,epchs,optimizer_bind,criterion_bind,dev)
    print('evaluating step 1 of pipeline, group by binding',flush=True)
    evaluate_binding(test_bind_loader,ligand_encoder,protein_encoder,0.05,dev)

    torch.save(ligand_encoder.state_dict(), 'ligand_encoder_pretrain.pth')
    torch.save(protein_encoder.state_dict(), 'protein_encoder_pretrain.pth')

    criterion_mech = nn.TripletMarginLoss(margin=1.0,p=2) ##triplet loss to start, couple w classfication maybe? not sure
    optimizer_mech = optim.AdamW(
        filter(lambda p: p.requires_grad, ligand_encoder.parameters()),
        lr=1e-4,
        weight_decay=1e-3)

    protein_encoder.eval()
    for p in protein_encoder.parameters():
        p.requires_grad_(False)

    ligand_encoder.train()
    mech_encoder_train(train_mech_loader,val_mech_loader,ligand_encoder,protein_encoder, epchs,optimizer_mech,criterion_mech,dev)
    print("evaluating test set, mechanism set",flush=True)

    evaluate_mechanism(test_mech_loader,ligand_encoder,protein_encoder,dev)

    torch.save(ligand_encoder.state_dict(), 'ligand_encoder_mech_final.pth')
    print('encodings made',flush=True)

if __name__ == "__main__":
    main()
