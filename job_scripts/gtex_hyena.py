import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
import gc
import os
import glob
import sys



project_dir = "Path/to/stored/datasets"
input_dir = f"{project_dir}/long_seqs_by_gene/"
output_dir = f"{project_dir}/embed_by_gene/cadph_long/"
checkpoint = "Path/to/the/downloaded/model/checkpoint"

max_length = 6001  # The maximum length after tokenization. For HyenaDNA, this is seq_length + 1.
batch_size = 128
seq_length = 6000  # The sequence length used in this experiment
files_per_job = 200  # Number of files to process per job


if len(sys.argv) > 1:
    job_index = int(sys.argv[1])
else:
    print("Error: Job index must be provided as a command line argument")
    exit(1)


# Get total number of available batch files
all_batch_files = glob.glob(f"{input_dir}batch_*.csv")
total_files = len(all_batch_files)
print(f"Total batch files available: {total_files}")

# Calculate start and end indices for this job
start_file = (job_index - 1) * files_per_job + 1
end_file = min(job_index * files_per_job, total_files)

# Check if job_index exceeds limit
if start_file > total_files:
    print(f"Error: job_index {job_index} exceeds the available number of files ({total_files})")
    exit(0)

print(f"Processing batch_{start_file}.csv to batch_{end_file}.csv")


os.makedirs(output_dir, exist_ok=True)
output_csv = f"{output_dir}embed_hyena_{start_file}_to_{end_file}.csv"

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModel.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    local_files_only=True
)
model.to(device)
model.eval()


class GTexDataset(Dataset):
    def __init__(self, df, seq_length=seq_length):
        super().__init__()
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


def mean_pooling(embed, attention_mask):
    """
    embed shape: [B, seq_len, 256]
    attention_mask shape: [B, seq_len]
    We'll do:
      embed = sum(embed * mask) / sum(mask)
    """
    attention_mask = attention_mask.unsqueeze(-1)
    embed = torch.sum(attention_mask * embed, dim=1) / torch.sum(attention_mask, dim=1)
    return embed


def process_files(start_idx, end_idx):
    all_embeds = []
    all_subj = []
    all_gene = []
    
    # Process each file in the range
    for file_idx in range(start_idx, end_idx + 1):
        input_csv = f"{input_dir}batch_{file_idx}.csv"
        print(f"Processing {input_csv}")
        
        try:
            df = pd.read_csv(input_csv)
            print(f"Loaded {input_csv} with {len(df)} rows")
            
            dataset = GTexDataset(df, seq_length=seq_length)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            del df
            gc.collect()
            
            # Process batches
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i % 10 == 0:
                        print(f"File {file_idx}/{end_idx}: Processing batch {i}/{len(dataloader)}")
                    
                    tok1 = tokenizer(
                        batch["allele1"],
                        padding='max_length',
                        truncation=True,
                        max_length=max_length,
                        return_tensors='pt'
                    )
                    tok1 = {k: v.to(device) for k, v in tok1.items()}
                    attn_mask1 = (tok1["input_ids"] != tokenizer.pad_token_id).int()
                    out1 = model(tok1["input_ids"])[0]
                    emb1 = mean_pooling(out1, attn_mask1)
                    
                    tok2 = tokenizer(
                        batch["allele2"],
                        padding='max_length',
                        truncation=True,
                        max_length=max_length,
                        return_tensors='pt'
                    )
                    tok2 = {k: v.to(device) for k, v in tok2.items()}
                    attn_mask2 = (tok2["input_ids"] != tokenizer.pad_token_id).int()
                    out2 = model(tok2["input_ids"])[0]
                    emb2 = mean_pooling(out2, attn_mask2)
                    
                    # Average Allele1 + Allele2
                    emb_avg = (emb1 + emb2) / 2.0
                    
                    all_embeds.append(emb_avg.cpu())
                    all_subj.extend(batch["subject_id"])
                    all_gene.extend(batch["gene_id"])
                    
                    del tok1, tok2, out1, out2, emb1, emb2
                    torch.cuda.empty_cache()
                    gc.collect()
        
        except Exception as e:
            print(f"Error processing file {input_csv}: {str(e)}")
            continue
    
    if len(all_embeds) == 0:
        return None
        
    # Stack all embeddings
    all_embeds = torch.cat(all_embeds, dim=0).numpy()
    
    hidden_dim = all_embeds.shape[1]
    emb_cols = [f"embedding_{i}" for i in range(hidden_dim)]
    df_out = pd.DataFrame(all_embeds, columns=emb_cols)
    
    df_out["subject_id"] = all_subj
    df_out["gene_id"] = all_gene
    
    df_out = df_out[["subject_id", "gene_id"] + emb_cols]
    return df_out


embed_df = process_files(start_file, end_file)

if embed_df is not None:
    embed_df.to_csv(output_csv, index=False)
    print(f"Embeddings saved to {output_csv} | Shape: {embed_df.shape}")
else:
    print("No embeddings were generated.")
