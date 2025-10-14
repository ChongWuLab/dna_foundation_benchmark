from enformer_pytorch import from_pretrained, seq_indices_to_one_hot
import pandas as pd
import torch
import gc
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import sys
import glob


project_dir = "Path/to/stored/datasets"
input_dir = f"{project_dir}/long_seqs_by_gene/"
output_dir = f"{project_dir}/embed_by_gene/cadph_long/"
checkpoint = "Path/to/the/downloaded/model/checkpoint"

batch_size = 16
seq_length = 196608
files_per_job = 2  # Process 2 files per job


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


max_job_index = (total_files + files_per_job - 1) // files_per_job
if job_index > max_job_index:
    print(f"Error: job_index {job_index} exceeds the maximum allowed ({max_job_index})")
    exit(0)

print(f"Processing batch_{start_file}.csv to batch_{end_file}.csv")


os.makedirs(output_dir, exist_ok=True)
output_csv = f"{output_dir}embed_enformer_{start_file}_to_{end_file}.csv"


device = torch.device("cuda")
model = from_pretrained(
    checkpoint,
    local_files_only=True
)
model.to(device)
model.eval()


def seq_to_indices(seq: str) -> torch.LongTensor:
    """
    Converts a DNA sequence (str) into a tensor of indices.
    Mapping: A->0, C->1, G->2, T->3, N and any other char ->4.
    """
    # Convert sequence to uppercase and then to ASCII bytes
    seq_bytes = seq.upper().encode("ascii")
    arr = np.frombuffer(seq_bytes, dtype=np.uint8)
    # Subtract 65 to map 'A'-'Z' into 0-25 (non-alphabetical chars will fall out of this range)
    arr = arr - 65


    lut = np.full(26, 4, dtype=np.int64)
    # Map A, C, G, T, N to 0,1,2,3,4 respectively
    for base, idx in zip("ACGTN", range(5)):
        lut[ord(base) - 65] = idx

    out = np.full(arr.shape, 4, dtype=np.int64)

    valid = (arr >= 0) & (arr < 26)
    out[valid] = lut[arr[valid]]
    return torch.from_numpy(out)


class GTexDataset(Dataset):
    def __init__(self, df, seq_length=seq_length):
        super().__init__()
        df["allele_1"] = df["allele_1"].str.upper()
        df["allele_2"] = df["allele_2"].str.upper()
        self.df = df.reset_index(drop=True)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.df)
    
    def _center_crop_and_convert_to_indices(self, sequence):
        """
        Extract a centered segment of the sequence.
        If the sequence is shorter than seq_length, pad with -1.
        Then convert the (cropped or padded) sequence to indices using vectorized conversion.
        """
        seq_len = len(sequence)
        if seq_len >= self.seq_length:
            # If the sequence is long enough, center crop
            start = (seq_len - self.seq_length) // 2
            cropped = sequence[start:start + self.seq_length]
            indices = seq_to_indices(cropped)
        else:
            # If sequence is too short, convert the full sequence first
            indices = seq_to_indices(sequence)
            # Calculate total padding required
            pad_total = self.seq_length - indices.shape[0]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            # Pad with -1 (padding value) on both sides symmetrically
            left_pad = torch.full((pad_left,), -1, dtype=torch.long)
            right_pad = torch.full((pad_right,), -1, dtype=torch.long)
            indices = torch.cat([left_pad, indices, right_pad])
        return indices

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        allele1_indices = self._center_crop_and_convert_to_indices(row["allele_1"])
        allele2_indices = self._center_crop_and_convert_to_indices(row["allele_2"])
        
        return {
            "allele1": allele1_indices,
            "allele2": allele2_indices,
            "subject_id": row["subject_id"],
            "gene_id": row["gene_id"],
        }

def collate_fn(batch_list):
    allele1_list = [item["allele1"] for item in batch_list]
    allele2_list = [item["allele2"] for item in batch_list]
    subj_list = [item["subject_id"] for item in batch_list]
    gene_list = [item["gene_id"] for item in batch_list]

    allele1_tensor = torch.stack(allele1_list)
    allele2_tensor = torch.stack(allele2_list)
    
    return {
        "allele1": allele1_tensor,
        "allele2": allele2_tensor,
        "subject_id": subj_list,
        "gene_id": gene_list,
    }


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
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=collate_fn
            )
            
            del df
            gc.collect()
            
            # Process batches
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i % 5 == 0:
                        print(f"File {file_idx}/{end_idx}: Processing batch {i}/{len(dataloader)}")
                    
                    allele1_one_hot = seq_indices_to_one_hot(batch["allele1"]).to(device, non_blocking=True)
                    allele2_one_hot = seq_indices_to_one_hot(batch["allele2"]).to(device, non_blocking=True)

                    _, embeddings1 = model(allele1_one_hot, return_embeddings=True)
                    _, embeddings2 = model(allele2_one_hot, return_embeddings=True)

                    mean_emb1 = embeddings1.mean(dim=1)
                    mean_emb2 = embeddings2.mean(dim=1)

                    # Average the two allele embeddings
                    emb_avg = (mean_emb1 + mean_emb2) / 2.0

                    all_embeds.append(emb_avg.cpu())
                    all_subj.extend(batch["subject_id"])
                    all_gene.extend(batch["gene_id"])

                    del allele1_one_hot, allele2_one_hot, embeddings1, embeddings2, mean_emb1, mean_emb2, emb_avg
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

















