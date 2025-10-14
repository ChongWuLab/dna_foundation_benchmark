import pandas as pd
import torch
import gc
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


project_dir = ".."
csv_path    = f"{project_dir}/data_processed/pathogenic/sequences_pathogenic.csv"
checkpoint  = "Path/to/the/downloaded/model/checkpoint"

max_length  = 1500
batch_size  = 64
output_csv  = f"{project_dir}/results_final/dnabert2_meanpool/pathogenic.csv"


df = pd.read_csv(csv_path)
print(f"Loaded {df.shape[0]} rows from {csv_path}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model.eval()
model.to(device)

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
            "ref_seq": row["ref_seq"],
            "alt_seq": row["alt_seq"],
            "class": row["class"],
            "chromosome": row["chromosome"],
            "ref": row["ref"],
            "pos": row["pos"],
            "alt": row["alt"],
            "MAF": row["MAF"],
            "split": row["split"],
            "label": row["label"]
        }

def collate_fn(batch):
    ref_list  = [item["ref_seq"] for item in batch]
    alt_list  = [item["alt_seq"] for item in batch]
    class_list= [item["class"]   for item in batch]
    chrom_list= [item["chromosome"] for item in batch]
    ref_allele= [item["ref"]    for item in batch]
    pos_list  = [item["pos"]    for item in batch]
    alt_allele= [item["alt"]    for item in batch]
    maf_list  = [item["MAF"]    for item in batch]
    split_list= [item["split"]  for item in batch]
    label_list= [item["label"]  for item in batch]
    
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


def mean_pooling(hidden_states, attention_mask):
    """
    hidden_states: [B, seq_len, hidden_dim]
    attention_mask: [B, seq_len]
    Return shape: [B, hidden_dim]
    """
    attention_mask = attention_mask.unsqueeze(-1).to(device)
    embed = torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
    return embed


# Distances
def l1_norm(e1, e2):
    return torch.norm(e1 - e2, p=1, dim=1)

def l2_norm(e1, e2):
    return torch.norm(e1 - e2, p=2, dim=1)

def cosine_similarity(e1, e2):
    return F.cosine_similarity(e1, e2, dim=1)

def dot_product(e1, e2):
    return torch.sum(e1 * e2, dim=1)



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
    for batch_idx, batch in enumerate(loader):
        tok_ref = tokenizer(
            batch["ref_seq"],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        tok_ref = {k: v.to(device) for k, v in tok_ref.items()}
        out_ref = model(tok_ref["input_ids"], attention_mask=tok_ref["attention_mask"])[0]
        emb_ref = mean_pooling(out_ref, tok_ref["attention_mask"])

        tok_alt = tokenizer(
            batch["alt_seq"],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        tok_alt = {k: v.to(device) for k, v in tok_alt.items()}
        out_alt = model(tok_alt["input_ids"], attention_mask=tok_alt["attention_mask"])[0]
        emb_alt = mean_pooling(out_alt, tok_alt["attention_mask"])

        emb_ref = emb_ref.cpu()
        emb_alt = emb_alt.cpu()


        l1_vals  = l1_norm(emb_ref, emb_alt).numpy()
        l2_vals  = l2_norm(emb_ref, emb_alt).numpy()
        cos_vals = cosine_similarity(emb_ref, emb_alt).numpy()
        dot_vals = dot_product(emb_ref, emb_alt).numpy()


        batch_size_curr = len(batch["class"])
        for i in range(batch_size_curr):
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

        del emb_ref, emb_alt, out_ref, out_alt
        torch.cuda.empty_cache()
        gc.collect()


df_out = pd.DataFrame({
    "L1": results["L1"],
    "L2": results["L2"],
    "cos": results["cos"],
    "dot": results["dot"],
    "class": results["class"],
    "chromosome": results["chromosome"],
    "ref": results["ref"],
    "pos": results["pos"],
    "alt": results["alt"],
    "MAF": results["MAF"],
    "split": results["split"],
    "label": results["label"]
})

df_out.to_csv(output_csv, index=False)
print(f"Saved => {output_csv} | Shape: {df_out.shape}")
