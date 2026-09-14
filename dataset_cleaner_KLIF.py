from opencadd.databases.klifs import setup_local, setup_remote
import pandas as pd
import requests
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def datacleaner_KLIF():
    session = setup_remote()
    df_structures = session.structures.all_structures()

    df_selected = df_structures[
        ["kinase.klifs_id",
            "ligand.klifs_id",
            "ligand_allosteric.klifs_id",
            "structure.dfg"
            ]].copy()

    df_selected = df_selected.rename(columns={"kinase.klifs_id": "kinase.name"})
    df_selected["ligand.type"] = np.where(
        df_selected["ligand_allosteric.klifs_id"].notna(), "allosteric", "orthosteric")

    ligands_df = session.ligands.all_ligands()
    ligand_to_smiles = (ligands_df.drop_duplicates(subset=["ligand.klifs_id"]).set_index("ligand.klifs_id")["ligand.smiles"].to_dict())
    df_selected["ligand.smiles"] = df_selected["ligand.klifs_id"].map(ligand_to_smiles)

    unique_kinase_ids = df_selected["kinase.name"].dropna().unique().tolist()
    kinases_df = session.kinases.by_kinase_klifs_id(unique_kinase_ids)
    kinase_to_uniprot = (kinases_df.drop_duplicates(subset=["kinase.klifs_id"]).set_index("kinase.klifs_id")["kinase.uniprot"].to_dict())
    df_selected["uniprot.id"] = df_selected["kinase.name"].map(kinase_to_uniprot)
    unique_uniprots = df_selected["uniprot.id"].dropna().unique()
    uniprot_to_full_sequence = {}
    for up_id in unique_uniprots:
        url = f"https://rest.uniprot.org/uniprotkb/{up_id}.fasta"
        response = requests.get(url,timeout=20)

        if response.status_code == 200:
            lines = response.text.split("\n")
            sequence = "".join([line.strip() for line in lines if not line.startswith(">")])
            uniprot_to_full_sequence[up_id] = sequence
        else:
            uniprot_to_full_sequence[up_id] = None

    df_selected["kinase.full_sequence"] = df_selected["uniprot.id"].map(uniprot_to_full_sequence)

    #get 85 sequence (AA about active site), stored for now (not currently used in phase 2)
    kinase_to_sequence = (kinases_df.drop_duplicates(subset=["kinase.klifs_id"]).set_index("kinase.klifs_id")["kinase.pocket"].to_dict())
    df_selected["kinase.85sequence"] = df_selected["kinase.name"].map(kinase_to_sequence)

    ##grab full sequenece of the protein
    df_selected = df_selected.dropna(subset=["ligand.smiles", "kinase.full_sequence", "kinase.85sequence"])

    mode_conditions = [
    df_selected["ligand.type"] == "allosteric",
    df_selected["structure.dfg"] == "in",
    df_selected["structure.dfg"] == "out",
    df_selected["structure.dfg"] == "out-like",
]
    mode_labels = [0, 1, 2, 3]  # 0 allosteric, 1 type1, 2 type2, 3 type1.5
    df_selected["binding_mode_label"] = np.select(mode_conditions, mode_labels, default=-1)
    df_selected = df_selected[df_selected["binding_mode_label"] != -1]
    return df_selected

def split_mech_dataset(mech_df,val_size=0.1,test_size=0.1,random_state=42,min_group_size=3):
    """split data by ligand identity, prevent data leakage"""
    gss_temp = GroupShuffleSplit(n_splits=1,test_size=(val_size+test_size),random_state=random_state)
    train_idx, temp_idx = next(gss_temp.split(mech_df,groups=mech_df["ligand.klifs_id"]))
    train_df = mech_df.iloc[train_idx].reset_index(drop=True)
    temp_df = mech_df.iloc[temp_idx].reset_index(drop=True)

    relative_test_size = test_size/(val_size+test_size)
    gss_val_test = GroupShuffleSplit(n_splits=1,test_size=relative_test_size,random_state=random_state)
    val_idx,test_idx = next(gss_val_test.split(temp_df,groups=temp_df["ligand.klifs_id"]))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df,val_df,test_df
