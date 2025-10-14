import pandas as pd
import torch
import gc
import os
import sys
import numpy as np
from torch.utils.data import DataLoader, Dataset
from enformer_pytorch import from_pretrained, seq_indices_to_one_hot


# Configuration

project_dir = ".."
input_csv_path = f"{project_dir}/data_processed/pathogenic/seqs_pathogenic_long.csv"
output_dir = f"{project_dir}/data_processed/pathogenic"

checkpoint = "Path/to/the/downloaded/model/checkpoint"
INPUT_SEQUENCE_LENGTH = 196608
BATCH_SIZE = 8


# Helper Functions

def seq_to_indices(seq: str) -> np.ndarray:
    """Converts a DNA sequence string to a numpy array of indices."""
    seq_bytes = seq.upper().encode("ascii")
    arr = np.frombuffer(seq_bytes, dtype=np.uint8)
    arr = arr - 65
    lut = np.full(26, 4, dtype=np.int8)
    for base, idx in zip("ACGT", range(4)):
        lut[ord(base) - 65] = idx
    out = np.full(arr.shape, 4, dtype=np.int8)
    valid = (arr >= 0) & (arr < 26)
    out[valid] = lut[arr[valid]]
    return out

def one_hot_encode_sequences(sequences: list[str], device: torch.device) -> torch.Tensor:
    """Converts a list of DNA strings to a one-hot encoded tensor on the specified device."""
    indices_list = [seq_to_indices(s) for s in sequences]
    indices_tensor = torch.from_numpy(np.stack(indices_list)).long()
    one_hot_tensor = seq_indices_to_one_hot(indices_tensor).to(device)
    return one_hot_tensor


def calculate_and_save_metrics(vectors1, vectors2, metadata_df, output_path, prefix, group_name):
    """
    Calculates metrics and saves them to CSV, including metadata in the output.
    """
    print(f"Calculating and saving metrics for prefix: {prefix}, group: {group_name}")
    os.makedirs(output_path, exist_ok=True)
    metadata_df = metadata_df.reset_index(drop=True)

    v1 = vectors1.cpu().numpy()
    v2 = vectors2.cpu().numpy()
    differences = v2 - v1
    l1 = np.sum(np.abs(differences), axis=1)
    l2 = np.linalg.norm(differences, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    dot_product = np.sum(v1 * v2, axis=1)
    cosine_sim = np.divide(dot_product, norm1 * norm2, out=np.zeros_like(dot_product, dtype=float), where=(norm1 * norm2) != 0)

    diff_df = pd.DataFrame(differences)
    # Concatenate side-by-side: [embedding_diff_0, ..., embedding_diff_N, chromosome, pos, ref, alt]
    output_diff_df = pd.concat([diff_df, metadata_df], axis=1)
    output_diff_df.to_csv(f"{output_path}/{prefix}_differences_{group_name}.csv", index=False, header=False)

    l1_output_df = metadata_df.copy()
    l1_output_df.insert(0, 'L1_distance', l1)
    l1_output_df.to_csv(f"{output_path}/{prefix}_L1_{group_name}.csv", index=False, header=True)

    l2_output_df = metadata_df.copy()
    l2_output_df.insert(0, 'L2_distance', l2)
    l2_output_df.to_csv(f"{output_path}/{prefix}_L2_{group_name}.csv", index=False, header=True)
    
    cosine_output_df = metadata_df.copy()
    cosine_output_df.insert(0, 'cosine_similarity', cosine_sim)
    cosine_output_df.to_csv(f"{output_path}/{prefix}_cosine_{group_name}.csv", index=False, header=True)

    print(f"Saved results to {output_path} with prefix '{prefix}' for group '{group_name}'")



class SequenceDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        df["ref_seq"] = df["ref_seq"].astype(str)
        df["alt_seq"] = df["alt_seq"].astype(str)
        self.df = df.reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()

def collate_fn(batch_list):
    """Collate function to handle all dataframe columns."""
    return {key: [item[key] for item in batch_list] for key in batch_list[0]}



if __name__ == "__main__":
    print("Loading Enformer model...")
    device = torch.device("cuda")
    model = from_pretrained(checkpoint, local_files_only=True)
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}.")

    # Load and Split Data
    print(f"\nLoading data from {input_csv_path}...")
    full_df = pd.read_csv(input_csv_path)

    print(f"Loaded {len(full_df)} total variants.")

    df_pathogenic = full_df[full_df['class'] == 1].copy()
    df_common = full_df[full_df['class'] == 0].copy()
    print(f"Pathogenic (class 1): {len(df_pathogenic)} variants")
    print(f"Common (class 0): {len(df_common)} variants")

    # Process Each Group
    datasets_to_process = {
        'pathogenic': df_pathogenic,
        'common': df_common
    }

    for group_name, df_group in datasets_to_process.items():
        print(f"PROCESSING GROUP: {group_name.upper()}")

        dataset = SequenceDataset(df_group)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        all_hidden_ref, all_hidden_alt = [], []
        all_outputs_ref, all_outputs_alt = [], []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                print(f"Processing batch {i+1}/{len(loader)} for group '{group_name}'...")
                
                ref_one_hot = one_hot_encode_sequences(batch["ref_seq"], device)
                alt_one_hot = one_hot_encode_sequences(batch["alt_seq"], device)
                
                ref_output, ref_hidden = model(ref_one_hot, return_embeddings=True)
                alt_output, alt_hidden = model(alt_one_hot, return_embeddings=True)

                hidden_vec_ref = ref_hidden.mean(dim=1)
                hidden_vec_alt = alt_hidden.mean(dim=1)
                all_hidden_ref.append(hidden_vec_ref.cpu())
                all_hidden_alt.append(hidden_vec_alt.cpu())

                output_vec_ref = ref_output['human'].mean(dim=1)
                output_vec_alt = alt_output['human'].mean(dim=1)
                all_outputs_ref.append(output_vec_ref.cpu())
                all_outputs_alt.append(output_vec_alt.cpu())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

        # Select the metadata columns to be saved from the group's dataframe.
        metadata_to_save = df_group[['chromosome', 'pos', 'ref', 'alt']]
        
        full_hidden_ref = torch.cat(all_hidden_ref, dim=0)
        full_hidden_alt = torch.cat(all_hidden_alt, dim=0)
        calculate_and_save_metrics(
            vectors1=full_hidden_ref,
            vectors2=full_hidden_alt,
            metadata_df=metadata_to_save,
            output_path=output_dir,
            prefix="enformer_hidden",
            group_name=group_name
        )
        
        full_outputs_ref = torch.cat(all_outputs_ref, dim=0)
        full_outputs_alt = torch.cat(all_outputs_alt, dim=0)
        calculate_and_save_metrics(
            vectors1=full_outputs_ref,
            vectors2=full_outputs_alt,
            metadata_df=metadata_to_save,
            output_path=output_dir,
            prefix="enformer_output",
            group_name=group_name
        )
    
    print("\nAll processing complete!")