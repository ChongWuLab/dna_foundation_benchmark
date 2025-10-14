import pandas as pd
import torch
import gc
import os
import sys
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

# Configuration
project_dir = ".."
input_csv_path = f"{project_dir}/data_processed/pathogenic/seqs_pathogenic_medium.csv"
output_dir = f"{project_dir}/data_processed/pathogenic"

checkpoint = "Path/to/the/downloaded/model/checkpoint"
INPUT_SEQUENCE_LENGTH = 6000
MAX_LENGTH = 1500
BATCH_SIZE = 64


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool hidden states using an attention mask."""
    attention_mask = attention_mask.unsqueeze(-1)
    embed = torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
    return embed

def get_dnabert_embedding(seq_list, tokenizer, model, max_length):
    """Embed sequences with a single forward pass using the DNABERT-2 model."""
    tok = tokenizer(seq_list, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    
    device = model.device
    tok = {k: v.to(device) for k, v in tok.items()}
    
    with torch.no_grad():
        hidden_states = model(tok["input_ids"], attention_mask=tok["attention_mask"])[0]

    embedding = mean_pooling(hidden_states, tok["attention_mask"])
    return embedding


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
    return {key: [item[key] for item in batch_list] for key in batch_list[0]}



if __name__ == "__main__":
    print("Loading DNABERT-2 tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}.")

    print(f"\nLoading data from {input_csv_path}...")
    full_df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(full_df)} total variants")

    df_pathogenic = full_df[full_df['class'] == 1].copy()
    df_common = full_df[full_df['class'] == 0].copy()
    print(f"Pathogenic (class 1): {len(df_pathogenic)} variants")
    print(f"Common (class 0): {len(df_common)} variants")

    datasets_to_process = {'pathogenic': df_pathogenic, 'common': df_common}

    for group_name, df_group in datasets_to_process.items():
        print(f"PROCESSING GROUP: {group_name.upper()}")

        dataset = SequenceDataset(df_group)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
        
        all_embeds_ref, all_embeds_alt = [], []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                print(f"Processing batch {i+1}/{len(loader)} for group '{group_name}'...")
                
                embed_ref = get_dnabert_embedding(batch["ref_seq"], tokenizer, model, MAX_LENGTH)
                embed_alt = get_dnabert_embedding(batch["alt_seq"], tokenizer, model, MAX_LENGTH)
                
                all_embeds_ref.append(embed_ref.cpu())
                all_embeds_alt.append(embed_alt.cpu())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        if not all_embeds_ref:
            print(f"No embeddings generated for group '{group_name}', skipping.")
            continue

        full_embeds_ref = torch.cat(all_embeds_ref, dim=0)
        full_embeds_alt = torch.cat(all_embeds_alt, dim=0)
        
        metadata_to_save = df_group[['chromosome', 'pos', 'ref', 'alt']]

        calculate_and_save_metrics(
            vectors1=full_embeds_ref,
            vectors2=full_embeds_alt,
            metadata_df=metadata_to_save,
            output_path=output_dir,
            prefix="dnabert2",
            group_name=group_name
        )
    
    print("\nAll processing complete!")