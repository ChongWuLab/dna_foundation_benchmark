from transformers import AutoTokenizer, AutoModelForMaskedLM

import pandas as pd
import numpy as np
import argparse
import torch
import time
from torch.utils.data import DataLoader, Dataset


checkpoint = "/rsrch4/home/biostatistics/hfeng3/.cache/huggingface/hub/models--InstaDeepAI--nucleotide-transformer-v2-500m-multi-species/snapshots/8ec6b8c3a7cd7ac89807039c106bb3067506674b"
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
parser.add_argument('--max_length',
                    type=int,
                    required=True,
                    help='The maximum sequence length to be put into model')
parser.add_argument('--pooling',
                    type=str,
                    choices=["cls", "mean"],
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
times = []
with torch.no_grad():
    for i, batch in enumerate(train_loader):

        start_time = time.time()

        x = tokenizer(batch["x"], padding='max_length', truncation=True, 
                      max_length=args.max_length, return_tensors="pt")
        
        y = batch["y"].float()
        
        attention_mask = x["input_ids"] != tokenizer.pad_token_id
        embed = model(x["input_ids"], 
                      attention_mask=attention_mask,
                      encoder_attention_mask=attention_mask,
                      output_hidden_states=True)['hidden_states'][-1]
        
        if args.pooling == "cls":
            embed = embed[:,0,:]
        else:
            attention_mask = torch.unsqueeze(attention_mask, dim=-1)
            embed = torch.sum(attention_mask*embed, axis=1)/torch.sum(attention_mask, axis=1)
        
        end_time = time.time()
        time_lapsed = end_time - start_time
        times.append(time_lapsed)
        embeddings.append(embed)
        targets.append(y)


mean_time = sum(times)/len(times)
embeddings = torch.cat(embeddings, 0).numpy()
targets = torch.cat(targets).unsqueeze(1).numpy()
data = np.concatenate([embeddings, targets], axis=1)
output = pd.DataFrame(data)
output.columns = [f"embedding_{i}" for i in range(1024)] + ["target"]
if args.pooling == "cls":
    output.to_csv(f'{args.data_path}/results/train_embed_ntv2.csv', index=False)
    with open(f'{args.data_path}/results/runtime_ntv2.txt', 'w') as file:
        file.write(str(mean_time))
else:
    output.to_csv(f'{args.data_path}/results/train_embed_ntv2_meanpool.csv', index=False)
    with open(f'{args.data_path}/results/runtime_ntv2_meanpool.txt', 'w') as file:
        file.write(str(mean_time))



# Testing data inference
embeddings = []
targets = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):

        x = tokenizer(batch["x"], padding='max_length', truncation=True, 
                      max_length=args.max_length, return_tensors="pt")
        
        y = batch["y"].float()
        
        attention_mask = x["input_ids"] != tokenizer.pad_token_id
        embed = model(x["input_ids"], 
                      attention_mask=attention_mask,
                      encoder_attention_mask=attention_mask,
                      output_hidden_states=True)['hidden_states'][-1]
        
        if args.pooling == "cls":
            embed = embed[:,0,:]
        else:
            attention_mask = torch.unsqueeze(attention_mask, dim=-1)
            embed = torch.sum(attention_mask*embed, axis=1)/torch.sum(attention_mask, axis=1)
        
        embeddings.append(embed)
        targets.append(y)

embeddings = torch.cat(embeddings, 0).numpy()
targets = torch.cat(targets).unsqueeze(1).numpy()
data = np.concatenate([embeddings, targets], axis=1)
output = pd.DataFrame(data)
output.columns = [f"embedding_{i}" for i in range(1024)] + ["target"]
if args.pooling == "cls":
    output.to_csv(f'{args.data_path}/results/test_embed_ntv2.csv', index=False)
else:
    output.to_csv(f'{args.data_path}/results/test_embed_ntv2_meanpool.csv', index=False)