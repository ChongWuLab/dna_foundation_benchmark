from transformers import AutoTokenizer, AutoModelForMaskedLM

import pandas as pd
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader, Dataset


project_dir = ".."
checkpoint = "Path/to/the/downloaded/model/checkpoint"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, 
                                          trust_remote_code=True,
                                          local_files_only=True)

print(f"Tokenizer: {tokenizer}")

model = AutoModelForMaskedLM.from_pretrained(checkpoint, 
                                   trust_remote_code=True,
                                   local_files_only=True)

model = model.to('cpu')
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
                    help='The maximum sequence length to be put into model')
parser.add_argument('--pooling',
                    type=str,
                    choices=["cls", "mean", "max"],
                    default="cls",
                    help='The pooling method of the output sequence of token embeddings')
args = parser.parse_args()

# Load the data
train_path = f"{args.data_path}/train.csv"
test_path = f"{args.data_path}/test.csv"
print(train_path, test_path)

train_data = SequenceDataset(pd.read_csv(train_path, header=0))
test_data = SequenceDataset(pd.read_csv(test_path, header=0))
train_loader = DataLoader(train_data, batch_size=256, shuffle=False)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)


# Training data inference
embeddings = []
targets = []
with torch.no_grad():
    for i, batch in enumerate(train_loader):

        x = tokenizer(batch["x"], padding='max_length', truncation=True, 
                      max_length=args.max_length, return_tensors="pt")

        y = batch["y"].float()
        
        attention_mask = x["attention_mask"]
        embed = model(x["input_ids"], 
                      attention_mask=attention_mask,
                      encoder_attention_mask=attention_mask,
                      output_hidden_states=True)['hidden_states'][-1]
        
        if args.pooling == "cls":
            embed = embed[:,0,:]
        elif args.pooling == "mean":
            attention_mask = torch.unsqueeze(attention_mask, dim=-1)
            embed = torch.sum(attention_mask*embed, axis=1)/torch.sum(attention_mask, axis=1)
        elif args.pooling == "max":
            attention_mask = torch.unsqueeze(attention_mask, dim=-1)
            embed, _ = torch.max(attention_mask*embed + (1 - attention_mask) * -1e9, dim=1)
        
        embeddings.append(embed)
        targets.append(y)


embeddings = torch.cat(embeddings, 0).numpy()
targets = torch.cat(targets).unsqueeze(1).numpy()
data = np.concatenate([embeddings, targets], axis=1)
output = pd.DataFrame(data)
output.columns = [f"embedding_{i}" for i in range(1024)] + ["target"]

os.makedirs(f"{project_dir}/embeddings/{args.data_name}", exist_ok=True)
if args.pooling == "cls":
    output.to_csv(f'{project_dir}/embeddings/{args.data_name}/train_embed_ntv2.csv', index=False)
elif args.pooling == "mean":
    output.to_csv(f'{project_dir}/embeddings/{args.data_name}/train_embed_ntv2_meanpool.csv', index=False)
else:
    output.to_csv(f'{project_dir}/embeddings/{args.data_name}/train_embed_ntv2_maxpool.csv', index=False)


# Testing data inference
embeddings = []
targets = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):

        x = tokenizer(batch["x"], padding='max_length', truncation=True, 
                      max_length=args.max_length, return_tensors="pt")
        
        y = batch["y"].float()
        
        attention_mask = x["attention_mask"]
        embed = model(x["input_ids"], 
                      attention_mask=attention_mask,
                      encoder_attention_mask=attention_mask,
                      output_hidden_states=True)['hidden_states'][-1]
        
        if args.pooling == "cls":
            embed = embed[:,0,:]
        elif args.pooling == "mean":
            attention_mask = torch.unsqueeze(attention_mask, dim=-1)
            embed = torch.sum(attention_mask*embed, axis=1)/torch.sum(attention_mask, axis=1)
        elif args.pooling == "max":
            attention_mask = torch.unsqueeze(attention_mask, dim=-1)
            embed, _ = torch.max(attention_mask*embed + (1 - attention_mask) * -1e9, dim=1)
        
        embeddings.append(embed)
        targets.append(y)

embeddings = torch.cat(embeddings, 0).numpy()
targets = torch.cat(targets).unsqueeze(1).numpy()
data = np.concatenate([embeddings, targets], axis=1)
output = pd.DataFrame(data)
output.columns = [f"embedding_{i}" for i in range(1024)] + ["target"]
if args.pooling == "cls":
    output.to_csv(f'{project_dir}/embeddings/{args.data_name}/test_embed_ntv2.csv', index=False)
elif args.pooling == "mean":
    output.to_csv(f'{project_dir}/embeddings/{args.data_name}/test_embed_ntv2_meanpool.csv', index=False)
else:
    output.to_csv(f'{project_dir}/embeddings/{args.data_name}/test_embed_ntv2_maxpool.csv', index=False)