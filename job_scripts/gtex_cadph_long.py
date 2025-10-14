import pandas as pd
import torch
import gc
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import os
import glob
import sys


project_dir = "Path/to/stored/datasets"
input_dir = f"{project_dir}/long_seqs_by_gene/"
output_dir = f"{project_dir}/embed_by_gene/cadph_long/"
checkpoint = "Path/to/the/downloaded/model/checkpoint"

max_length = 131072  # The maximum model length after tokenization. For Caduceus, this is seq_length + 1.
batch_size = 16
seq_length = 131072  # The sequence length used in this experiment
files_per_job = 5  # Number of files to process per job


# Get job index from command line argument
if len(sys.argv) > 1:
    job_index = int(sys.argv[1])
else:
    print("Error: Job index must be provided as a command line argument")
    exit(1)


# File range calculation
# Get total number of available batch files
all_batch_files = glob.glob(f"{input_dir}batch_*.csv")
total_files = len(all_batch_files)
print(f"Total batch files available: {total_files}")

# Calculate start and end indices for this job
start_file = (job_index - 1) * files_per_job + 1
end_file = min(job_index * files_per_job, total_files)

# Check if job_index is too large
if start_file > total_files:
    print(f"Error: job_index {job_index} exceeds the available number of files ({total_files})")
    exit(0)

print(f"Processing batch_{start_file}.csv to batch_{end_file}.csv")


os.makedirs(output_dir, exist_ok=True)
output_csv = f"{output_dir}embed_cadph_{start_file}_to_{end_file}.csv"


tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    local_files_only=True
)
model = AutoModelForMaskedLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    local_files_only=True,
    device_map="cuda"
)
model.eval()


def reverse_complement(sequence: str) -> str:
    complement_map = str.maketrans("ACGT", "TGCA")
    return sequence.translate(complement_map)[::-1]


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


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    hidden_states: [B, seq_len, hidden_dim=256]
    attention_mask: [B, seq_len]
    We'll do sum(...) / sum(mask).
    """
    # Expand mask to [B, seq_len, 1] to multiply by hidden_states
    attention_mask = attention_mask.unsqueeze(-1)     # [B, seq_len, 1]
    embed = torch.sum(attention_mask * hidden_states, dim=1) / torch.sum(attention_mask, dim=1)
    return embed


def embed_with_rc(seq_list, tokenizer, model, max_length):
    """
    1. For each sequence in seq_list, create forward sequence and RC.
    2. Forward pass each (last hidden state).
    3. Average forward + RC => final hidden_state for that allele.
    4. Mean pool across tokens => final [B, hidden_dim=256].
    """
    tok_fwd = tokenizer(seq_list, padding='max_length', truncation=True,
                        max_length=max_length, return_tensors='pt')
    # Reverse complement each sequence
    rc_list = [reverse_complement(seq) for seq in seq_list]

    tok_rc = tokenizer(rc_list, padding='max_length', truncation=True,
                       max_length=max_length, return_tensors='pt')


    device = next(model.parameters()).device
    tok_fwd = {k: v.to(device) for k, v in tok_fwd.items()}
    tok_rc  = {k: v.to(device) for k, v in tok_rc.items()}

    with torch.no_grad():
        hidden_fwd = model(tok_fwd["input_ids"], output_hidden_states=True).hidden_states[-1]
        hidden_rc  = model(tok_rc["input_ids"],  output_hidden_states=True).hidden_states[-1]


    hidden_avg = (hidden_fwd + hidden_rc) / 2.0

    attn_mask_fwd = (tok_fwd["input_ids"] != tokenizer.pad_token_id).int()

    # Mean pooling
    embed = mean_pooling(hidden_avg, attn_mask_fwd)

    return embed


def process_files(start_idx, end_idx):
    all_embeds = []
    all_subj = []
    all_gene = []
    
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
            
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i % 10 == 0:
                        print(f"File {file_idx}/{end_idx}: Processing batch {i}/{len(dataloader)}")
                    
                    # Allele1 with forward and reverse complement
                    emb1 = embed_with_rc(batch["allele1"], tokenizer, model, max_length)
                    
                    # Allele2 with forward and reverse complement
                    emb2 = embed_with_rc(batch["allele2"], tokenizer, model, max_length)
                    
                    # Average Allele1 + Allele2 => final vector
                    emb_avg = (emb1 + emb2) / 2.0
                    
                    all_embeds.append(emb_avg.cpu())
                    all_subj.extend(batch["subject_id"])
                    all_gene.extend(batch["gene_id"])
                    
                    del emb1, emb2
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
