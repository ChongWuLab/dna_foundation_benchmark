from transformers import AutoModelForMaskedLM, AutoTokenizer

import pandas as pd
import numpy as np
import argparse
import torch
import gc
from torch.utils.data import DataLoader, Dataset

project_dir = ".."
checkpoint = "Path/to/the/downloaded/model/checkpoint"
tokenizer = AutoTokenizer.from_pretrained(checkpoint,
                                          trust_remote_code=True,
                                          local_files_only=True)
model = AutoModelForMaskedLM.from_pretrained(checkpoint,
                                  trust_remote_code=True,
                                  local_files_only=True,
                                  device_map="auto")
model.eval()

class SequenceDataset(Dataset):
    
    def __init__(self, dataframe):
        super().__init__()
        self.df = dataframe
        
        self.df.iloc[:,0] = self.df.iloc[:,0].str.upper()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = {"x":self.df.iloc[idx, 0],
                "y":self.df.iloc[idx, 1]}
        return data
    

# Function to compute reverse complement
def reverse_complement(sequence):
    complement_map = str.maketrans("ACGT", "TGCA")
    return sequence.translate(complement_map)[::-1]

def clear_gpu_memory():
    """Helper function to clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# Load command line arguments
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--data_path', 
                    type=str, 
                    required=True,
                    help='The path of the dataset, specifically, the directory that the train.csv and test.csv lies in')
parser.add_argument('--data_name',
                    type=str,
                    required=True,
                    help='The name of dataset, used for storing results')
parser.add_argument('--max_length',
                    type=int,
                    required=True,
                    help='The maximum sequence length to be put into model for padding')
args = parser.parse_args()

# Load the data
train_path = f"{args.data_path}/train.csv"
test_path = f"{args.data_path}/test.csv"

train_data = SequenceDataset(pd.read_csv(train_path, header=0))
test_data = SequenceDataset(pd.read_csv(test_path, header=0))
train_loader = DataLoader(train_data, batch_size=128, shuffle=False)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


# Inference Function with RC Handling
def process_data(loader, model, tokenizer, max_length):
    embeddings_sep = []
    embeddings_mean = []
    embeddings_max = []
    targets = []

    with torch.no_grad():
        for i, batch in enumerate(loader):

            # Tokenize forward sequence
            x = tokenizer(batch["x"], padding='max_length', truncation=True,
                          max_length=max_length, return_tensors="pt")
            # Generate reverse complement sequences
            rc_sequences = [reverse_complement(seq) for seq in batch["x"]]
            rc_x = tokenizer(rc_sequences, padding='max_length', truncation=True,
                             max_length=max_length, return_tensors="pt")
            y = batch["y"].float()

            # Move data to GPU
            x = {k: v.to(model.device) for k, v in x.items()}
            rc_x = {k: v.to(model.device) for k, v in rc_x.items()}

            # Forward pass for input and RC sequences
            forward_embed = model(x["input_ids"],
                                  output_hidden_states=True,
                                  return_dict=True).hidden_states[-1]
            clear_gpu_memory()
            rc_embed = model(rc_x["input_ids"],
                             output_hidden_states=True,
                             return_dict=True).hidden_states[-1]
            clear_gpu_memory()

            # Average forward and RC embeddings
            embed = (forward_embed + rc_embed) / 2

            # Compute attention mask
            attention_mask = (x["input_ids"] != tokenizer.pad_token_id).int()
            attention_mask = torch.unsqueeze(attention_mask, dim=-1)

            # Pool embeddings
            embed_sep = embed[:, embed.shape[1]-1, :]  # SEP embedding
            embed_mean = torch.sum(attention_mask * embed, axis=1) / torch.sum(attention_mask, axis=1)  # Mean pooling
            embed_max, _ = torch.max(attention_mask * embed + (1 - attention_mask) * -1e9, dim=1)  # Max pooling

            # Move embeddings to CPU
            embed_sep = embed_sep.cpu()
            embed_mean = embed_mean.cpu()
            embed_max = embed_max.cpu()

            # Append results
            embeddings_sep.append(embed_sep)
            embeddings_mean.append(embed_mean)
            embeddings_max.append(embed_max)
            targets.append(y)
            del forward_embed, rc_embed, embed

    # Final outputs
    embeddings_sep = torch.cat(embeddings_sep, 0).numpy()
    embeddings_mean = torch.cat(embeddings_mean, 0).numpy()
    embeddings_max = torch.cat(embeddings_max, 0).numpy()
    targets = torch.cat(targets).unsqueeze(1).numpy()

    return embeddings_sep, embeddings_mean, embeddings_max, targets


# Process training data
train_embeddings_sep, train_embeddings_mean, train_embeddings_max, train_targets = process_data(
    train_loader, model, tokenizer, args.max_length
)



# Save training results
os.makedirs(f"{project_dir}/embeddings/{args.data_name}", exist_ok=True)

output = pd.DataFrame(np.concatenate([train_embeddings_sep, train_targets], axis=1))
output.columns = [f"embedding_{i}" for i in range(256)] + ["target"]
output.to_csv(f'{project_dir}/embeddings/{args.data_name}/train_embed_cadph.csv', index=False)

output = pd.DataFrame(np.concatenate([train_embeddings_mean, train_targets], axis=1))
output.columns = [f"embedding_{i}" for i in range(256)] + ["target"]
output.to_csv(f'{project_dir}/embeddings/{args.data_name}/train_embed_cadph_meanpool.csv', index=False)

output = pd.DataFrame(np.concatenate([train_embeddings_max, train_targets], axis=1))
output.columns = [f"embedding_{i}" for i in range(256)] + ["target"]
output.to_csv(f'{project_dir}/embeddings/{args.data_name}/train_embed_cadph_maxpool.csv', index=False)

del train_embeddings_sep, train_embeddings_mean, train_embeddings_max, train_targets
clear_gpu_memory()


# Process testing data
test_embeddings_sep, test_embeddings_mean, test_embeddings_max, test_targets = process_data(
    test_loader, model, tokenizer, args.max_length
)

# Save testing results
output = pd.DataFrame(np.concatenate([test_embeddings_sep, test_targets], axis=1))
output.columns = [f"embedding_{i}" for i in range(256)] + ["target"]
output.to_csv(f'{project_dir}/embeddings/{args.data_name}/test_embed_cadph.csv', index=False)

output = pd.DataFrame(np.concatenate([test_embeddings_mean, test_targets], axis=1))
output.columns = [f"embedding_{i}" for i in range(256)] + ["target"]
output.to_csv(f'{project_dir}/embeddings/{args.data_name}/test_embed_cadph_meanpool.csv', index=False)

output = pd.DataFrame(np.concatenate([test_embeddings_max, test_targets], axis=1))
output.columns = [f"embedding_{i}" for i in range(256)] + ["target"]
output.to_csv(f'{project_dir}/embeddings/{args.data_name}/test_embed_cadph_maxpool.csv', index=False)