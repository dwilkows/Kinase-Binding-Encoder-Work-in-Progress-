from chembl_webresource_client.new_client import new_client
import pandas as pd
from rdkit import Chem
import requests
from sklearn.model_selection import train_test_split
import time

def clean_dataset(target_chembl_ids=None):
    molecule = new_client.molecule
    activity = new_client.activity
    target = new_client.target

    def make_label(value,measurement_type):
        if measurement_type in ["Ki", "Kd"]:
            if value <= 1000:
                return 1
            elif value >= 10000:
                return 0
        elif measurement_type == "IC50":

            if value <= 500:
                return 1
            elif value >= 5000:
                return 0
        return None

    def get_sequence(uniprot_id):
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                return None

            fasta = r.text
            sequence = "".join(
                fasta.split("\n")[1:])
            return sequence
        except Exception:
            return None

    seq_cache = {}

    def chembl_to_sequence(target_id):
        if target_id in seq_cache:
            return seq_cache[target_id]
        try:
            target_info = target.get(target_id)
            components = target_info.get(
                "target_components",
                [])
            if len(components) == 0:
                seq_cache[target_id] = None
                return None
            accession = components[0].get("accession")
            if accession is None:
                seq_cache[target_id] = None
                return None
            seq = get_sequence(accession)
            seq_cache[target_id] = seq
            return seq
        except Exception:
            seq_cache[target_id] = None
            return None

    filter_kwargs = dict(assay_type="B", standard_type__in=["Ki","IC50","Kd"])
    if target_chembl_ids is not None:
        filter_kwargs["target_chembl_id__in"] = list(target_chembl_ids)

    activities = activity.filter(
        **filter_kwargs
    ).only(["molecule_chembl_id","target_chembl_id","standard_value","standard_units","standard_relation","standard_type"])

    rows = []
    start = time.time()
    for i, act in enumerate(activities):
        rows.append(act)
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  ...{i + 1} records fetched ({elapsed:.0f}s elapsed, ~{rate:.1f} rows/sec)")

    print(f"Finished fetching: {len(rows)} activity records in {time.time() - start:.0f}s")
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"ChEMBL query returned 0 activity records for filter={filter_kwargs}.")
    df["standard_value"] = pd.to_numeric(
        df["standard_value"],
        errors="coerce"
    )
    df = df.dropna(subset=["standard_value"])
    df = df[df["standard_units"] == "nM"]
    df = df[df["standard_relation"] == "="]
    df["label"] = df.apply(lambda row: make_label(row["standard_value"],row["standard_type"]),axis=1)

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df = df.groupby(["molecule_chembl_id","target_chembl_id"], as_index=False).agg(
       label=("label", lambda x: x.mode()[0] if x.nunique() == 1 else -1),  # -1 = conflicting, drop later
       standard_value=("standard_value","first"),
       standard_type=("standard_type","first"),
   )
    df = df[df["label"] != -1]
    smiles_data = molecule.filter(
        molecule_chembl_id__in=list(
            df["molecule_chembl_id"].unique())).only(
        ["molecule_chembl_id","molecule_structures","standard_type"])

    smiles_map = {}

    for mol in smiles_data:
        structures = mol.get("molecule_structures")
        if structures is None:
            continue
        smiles = structures.get("canonical_smiles")
        if smiles is None:
            continue
        smiles_map[mol["molecule_chembl_id"]] = smiles

    df["smiles"] = df["molecule_chembl_id"].map(smiles_map)

    df = df.dropna(subset=["smiles"])
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["mol"])

    unique_targets = (
        df["target_chembl_id"].unique())
    target_to_seq = {}

    for target_id in unique_targets:
        target_to_seq[target_id] = (chembl_to_sequence(target_id))
    df["sequence"] = df["target_chembl_id"].map(target_to_seq)
    df = df.dropna(subset=["sequence"])
    df = df[["smiles","sequence","label"]].reset_index(drop=True)

    return df

def split_bind_dataset(chem_df, val_size=0.1,test_size=0.1,random_state=42):
    train_df,temp_df = train_test_split(chem_df,test_size=(val_size+test_size),stratify=chem_df["label"],random_state=random_state)
    rel_test_size = test_size/(val_size + test_size)
    val_df,test_df = train_test_split(temp_df,test_size=rel_test_size,stratify=temp_df["label"],random_state=random_state)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df,val_df,test_df
