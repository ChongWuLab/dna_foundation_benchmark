import pandas as pd
import torch
import gc
import os
import sys
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.nn.functional as F


# Configuration

project_dir = ".."
input_csv_path = f"{project_dir}/data_processed/pathogenic/seqs_pathogenic_long.csv"
output_dir = f"{project_dir}/data_processed/pathogenic"

checkpoint = "Path/to/the/downloaded/model/checkpoint"
INPUT_SEQUENCE_LENGTH = 196608
MAX_LENGTH = 131072
BATCH_SIZE = 16


# Helper Functions

def get_center_sequence(full_sequence: str, target_length: int) -> str:
    """Extracts the central part of a sequence."""
    full_len = len(full_sequence)
    
    start = (full_len - target_length) // 2
    end = start + target_length
    return full_sequence[start:end]

def reverse_complement(sequence: str) -> str:
    """Reverse complement a DNA sequence."""
    complement_map = str.maketrans("ACGTN", "TGCAN")
    return sequence.upper().translate(complement_map)[::-1]

def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool hidden states using an attention mask."""
    attention_mask = attention_mask.unsqueeze(-1)
    embed = torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
    return embed

def embed_with_rc(seq_list, tokenizer, model, max_length):
    """Embed sequences by averaging forward and reverse-complement passes."""
    tok_fwd = tokenizer(seq_list, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    rc_list = [reverse_complement(seq) for seq in seq_list]
    tok_rc = tokenizer(rc_list, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    
    device = model.device
    tok_fwd = {k: v.to(device) for k, v in tok_fwd.items()}
    tok_rc  = {k: v.to(device) for k, v in tok_rc.items()}
    
    with torch.no_grad():
        out_fwd = model(tok_fwd["input_ids"], output_hidden_states=True).hidden_states[-1]
        out_rc  = model(tok_rc["input_ids"],  output_hidden_states=True).hidden_states[-1]
    
    hidden_avg = (out_fwd + out_rc) / 2.0
    attn_mask = (tok_fwd["input_ids"] != tokenizer.pad_token_id).int()
    embed_allele = mean_pooling(hidden_avg, attn_mask)
    return embed_allele


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



# Dataset and Collation

class SequenceDataset(Dataset):
    def __init__(self, df, target_length):
        super().__init__()
        df["ref_seq"] = df["ref_seq"].astype(str)
        df["alt_seq"] = df["alt_seq"].astype(str)
        self.df = df.reset_index(drop=True)
        self.target_length = target_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get the full-length sequences
        row_dict = self.df.iloc[idx].to_dict()
        
        # Perform center-cropping
        row_dict["ref_seq"] = get_center_sequence(row_dict["ref_seq"], self.target_length)
        row_dict["alt_seq"] = get_center_sequence(row_dict["alt_seq"], self.target_length)
        
        return row_dict

def collate_fn(batch_list):
    return {key: [item[key] for item in batch_list] for key in batch_list[0]}


# Main Execution

if __name__ == "__main__":
    print("Loading Caduceus-Ph tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True, device_map="auto")
    model.eval()
    print("Model loaded successfully.")

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

        dataset = SequenceDataset(df_group, target_length=MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
        
        all_embeds_ref, all_embeds_alt = [], []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                print(f"Processing batch {i+1}/{len(loader)} for group '{group_name}'...")

                embed_ref = embed_with_rc(batch["ref_seq"], tokenizer, model, MAX_LENGTH)
                embed_alt = embed_with_rc(batch["alt_seq"], tokenizer, model, MAX_LENGTH)
                
                all_embeds_ref.append(embed_ref.cpu()); all_embeds_alt.append(embed_alt.cpu())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache(); gc.collect()

            
        full_embeds_ref = torch.cat(all_embeds_ref, dim=0)
        full_embeds_alt = torch.cat(all_embeds_alt, dim=0)
        
        
        metadata_to_save = df_group[['chromosome', 'pos', 'ref', 'alt']]

        calculate_and_save_metrics(
            vectors1=full_embeds_ref,
            vectors2=full_embeds_alt,
            metadata_df=metadata_to_save,
            output_path=output_dir,
            prefix="cadph_long",
            group_name=group_name
        )

    print("All processing complete!")