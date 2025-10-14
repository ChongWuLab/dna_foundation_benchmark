import pandas as pd
import torch
import gc
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.nn.functional as F


project_dir = ".."
csv_path = f"{project_dir}/data_processed/pathogenic/sequences_pathogenic.csv"
checkpoint  = "Path/to/the/downloaded/model/checkpoint"
max_length  = 6001
batch_size  = 64
output_csv  = f"{project_dir}/results_final/cadph_meanpool/pathogenic.csv"


df = pd.read_csv(csv_path)
print(f"Loaded {df.shape[0]} rows from {csv_path}")


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


# Dataset + DataLoader
class PathogenicDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        df["ref_seq"] = df["ref_seq"].str.upper()
        df["alt_seq"] = df["alt_seq"].str.upper()
        self.df = df.reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "ref_seq":  row["ref_seq"],
            "alt_seq":  row["alt_seq"],
            "class":    row["class"],
            "chromosome": row["chromosome"],
            "ref":      row["ref"],
            "pos":      row["pos"],
            "alt":      row["alt"],
            "MAF":      row["MAF"] if "MAF" in self.df.columns else None,
            "split":    row["split"],
            "label":    row["label"]
        }

def collate_fn(batch_list):
    ref_list  = [item["ref_seq"] for item in batch_list]
    alt_list  = [item["alt_seq"] for item in batch_list]
    class_list= [item["class"]   for item in batch_list]
    chrom_list= [item["chromosome"] for item in batch_list]
    ref_allele= [item["ref"]    for item in batch_list]
    pos_list  = [item["pos"]    for item in batch_list]
    alt_allele= [item["alt"]    for item in batch_list]
    maf_list  = [item["MAF"]    for item in batch_list]
    split_list= [item["split"]  for item in batch_list]
    label_list= [item["label"]  for item in batch_list]
    
    return {
        "ref_seq":  ref_list,
        "alt_seq":  alt_list,
        "class":    class_list,
        "chromosome": chrom_list,
        "ref":      ref_allele,
        "pos":      pos_list,
        "alt":      alt_allele,
        "MAF":      maf_list,
        "split":    split_list,
        "label":    label_list
    }

dataset = PathogenicDataset(df)
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# Mean Pooling Helper
def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # hidden_states: [B, seq_len, hidden_dim=256]
    # attention_mask: [B, seq_len]
    # We'll do sum(...) / sum(mask)
    attention_mask = attention_mask.unsqueeze(-1)
    embed = torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
    return embed


# 6. embed_with_rc: Forward + RC => average => mean pool
def embed_with_rc(seq_list, tokenizer, model, max_length):
    """
    1) For each sequence in seq_list, do forward pass + reverse complement pass.
    2) Average the two hidden states => final hidden_state for that sequence.
    3) Then mean pool => [B, 256].
    """
    # Forward sequences
    tok_fwd = tokenizer(
        seq_list,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    # Reverse complement each
    rc_list = [reverse_complement(seq) for seq in seq_list]
    tok_rc = tokenizer(
        rc_list,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    device = model.device
    tok_fwd = {k: v.to(device) for k, v in tok_fwd.items()}
    tok_rc  = {k: v.to(device) for k, v in tok_rc.items()}
    
    with torch.no_grad():
        out_fwd = model(tok_fwd["input_ids"], output_hidden_states=True).hidden_states[-1]  # [B, seq_len, 256]
        out_rc  = model(tok_rc["input_ids"],  output_hidden_states=True).hidden_states[-1]  # [B, seq_len, 256]
    
    # Average forward + RC => shape [B, seq_len, 256]
    hidden_avg = (out_fwd + out_rc) / 2.0
    
    # attention mask for forward
    attn_mask = (tok_fwd["input_ids"] != tokenizer.pad_token_id).int()
    
    # mean pool => shape [B, 256]
    embed_allele = mean_pooling(hidden_avg, attn_mask)
    return embed_allele


# Distance / Similarity Functions
def l1_norm(e_ref, e_alt):
    return torch.norm(e_ref - e_alt, p=1, dim=1)

def l2_norm(e_ref, e_alt):
    return torch.norm(e_ref - e_alt, p=2, dim=1)

def cosine_similarity(e_ref, e_alt):
    return F.cosine_similarity(e_ref, e_alt, dim=1)

def dot_product(e_ref, e_alt):
    return torch.sum(e_ref * e_alt, dim=1)


# Inference
results = {
    "L1": [],
    "L2": [],
    "cos": [],
    "dot": [],
    "class": [],
    "chromosome": [],
    "ref": [],
    "pos": [],
    "alt": [],
    "MAF": [],
    "split": [],
    "label": []
}

with torch.no_grad():
    for batch in loader:
        embed_ref = embed_with_rc(batch["ref_seq"], tokenizer, model, max_length)
        
        embed_alt = embed_with_rc(batch["alt_seq"], tokenizer, model, max_length)
        
        embed_ref = embed_ref.cpu()
        embed_alt = embed_alt.cpu()
        
        l1_vals  = l1_norm(embed_ref, embed_alt).numpy()
        l2_vals  = l2_norm(embed_ref, embed_alt).numpy()
        cos_vals = cosine_similarity(embed_ref, embed_alt).numpy()
        dot_vals = dot_product(embed_ref, embed_alt).numpy()
        
        for i in range(len(batch["class"])):
            results["L1"].append(l1_vals[i])
            results["L2"].append(l2_vals[i])
            results["cos"].append(cos_vals[i])
            results["dot"].append(dot_vals[i])
            
            results["class"].append(batch["class"][i])
            results["chromosome"].append(batch["chromosome"][i])
            results["ref"].append(batch["ref"][i])
            results["pos"].append(batch["pos"][i])
            results["alt"].append(batch["alt"][i])
            results["MAF"].append(batch["MAF"][i])
            results["split"].append(batch["split"][i])
            results["label"].append(batch["label"][i])
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


df_out = pd.DataFrame({
    "L1":   results["L1"],
    "L2":   results["L2"],
    "cos":  results["cos"],
    "dot":  results["dot"],
    "class":results["class"],
    "chromosome": results["chromosome"],
    "ref":  results["ref"],
    "pos":  results["pos"],
    "alt":  results["alt"],
    "MAF":  results["MAF"],
    "split":results["split"],
    "label":results["label"]
})

df_out.to_csv(output_csv, index=False)
print(f"Saved => {output_csv} | Shape: {df_out.shape}")
