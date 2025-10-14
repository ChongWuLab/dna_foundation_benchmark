import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
import gc
import os
import sys


project_dir = "Path/to/stored/datasets"
input_dir = f"{project_dir}/long_seqs_by_gene/"
output_dir = f"{project_dir}/embed_by_gene/cadph_long/"
checkpoint = "Path/to/the/downloaded/model/checkpoint"

max_length = 1500  # The maximum model length after tokenization
batch_size = 128
seq_length = 6000  # The sequence length used in this experiment


job_index = int(os.environ["LSB_JOBINDEX"])
print(f"Processing batch_{job_index}.csv")

input_csv = f"{input_dir}batch_{job_index}.csv"
output_csv = f"{output_dir}embed_dnabert2_{job_index}.csv"

os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv(input_csv)
print(f"Loaded {input_csv} with {len(df)} rows")


device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(checkpoint, 
                                          trust_remote_code=True,
                                          local_files_only=True)
model = AutoModel.from_pretrained(checkpoint, 
                                  trust_remote_code=True,
                                  local_files_only=True)
model.eval()
model.to(device)


class GTexDataset(Dataset):
    def __init__(self, df, seq_length=seq_length):
        super().__init__()
        # Ensure uppercase
        df["allele_1"] = df["allele_1"].str.upper()
        df["allele_2"] = df["allele_2"].str.upper()
        self.df = df.reset_index(drop=True)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.df)
    
    def take_center_seq(self, s, n):
        start = (len(s) - n) // 2
        return s[start:start + n] if start >= 0 else s[:n]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Take the center base pairs set by seq_length
        allele1 = self.take_center_seq(row["allele_1"], self.seq_length)
        allele2 = self.take_center_seq(row["allele_2"], self.seq_length)
        return {
            "allele1": allele1,
            "allele2": allele2,
            "subject_id": row["subject_id"],
            "gene_id": row["gene_id"],
        }

def collate_fn(batch_list):
    allele1_list = [item["allele1"] for item in batch_list]
    allele2_list = [item["allele2"] for item in batch_list]
    subj_list    = [item["subject_id"] for item in batch_list]
    gene_list    = [item["gene_id"] for item in batch_list]

    return {
        "allele1": allele1_list,
        "allele2": allele2_list,
        "subject_id": subj_list,
        "gene_id": gene_list,
    }

dataset = GTexDataset(df)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


del df
gc.collect()


def mean_pooling(model_output, attention_mask):
    attention_mask = torch.unsqueeze(attention_mask, dim=-1)  # [B, seq_len, 1]
    embed = torch.sum(attention_mask * model_output, dim=1) / torch.sum(attention_mask, dim=1)
    return embed


def generate_embeddings(data_loader):
    all_embeds = []
    all_subj   = []
    all_gene   = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(data_loader)}")
            
            tok1 = tokenizer(batch["allele1"], 
                             padding='max_length',
                             truncation=True,
                             max_length=max_length,
                             return_tensors='pt')
            out1 = model(tok1["input_ids"], attention_mask=tok1["attention_mask"])[0]
            emb1 = mean_pooling(out1, tok1["attention_mask"])

            tok2 = tokenizer(batch["allele2"], 
                             padding='max_length',
                             truncation=True,
                             max_length=max_length,
                             return_tensors='pt')
            out2 = model(tok2["input_ids"], attention_mask=tok2["attention_mask"])[0]
            emb2 = mean_pooling(out2, tok2["attention_mask"])

            # Average Allele1 and Allele2
            emb_avg = (emb1 + emb2) / 2.0

            all_embeds.append(emb_avg.detach().cpu())
            all_subj.extend(batch["subject_id"])
            all_gene.extend(batch["gene_id"])

    # Stack
    all_embeds = torch.cat(all_embeds, dim=0).numpy()

    hidden_dim = all_embeds.shape[1]
    emb_cols   = [f"embedding_{i}" for i in range(hidden_dim)]
    df_out     = pd.DataFrame(all_embeds, columns=emb_cols)
    
    df_out["subject_id"] = all_subj
    df_out["gene_id"] = all_gene

    df_out = df_out[["subject_id", "gene_id"] + emb_cols]
    return df_out

embed_df = generate_embeddings(dataloader)
embed_df.to_csv(output_csv, index=False)
print(f"Embeddings saved to {output_csv} | Shape: {embed_df.shape}")
