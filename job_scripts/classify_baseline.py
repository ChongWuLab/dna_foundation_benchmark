import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, accuracy_score
import os

project_dir = ".."

# Simple CNN architecture for DNA sequence classification
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleCNN, self).__init__()
        
        # Input channels = 5 (A, C, G, T, N)
        self.conv1 = nn.Conv1d(5, 64, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Global pooling to handle variable length sequences
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)
        
        # Activation for binary classification
        self.sigmoid = nn.Sigmoid()
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Input shape: (batch_size, 5, seq_length)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        
        # Apply sigmoid only for binary classification
        if x.size(1) == 1:
            x = self.sigmoid(x)
        
        return x

# Dataset class for DNA sequences
class SequenceDataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.df = dataframe
        
        # Convert sequences to uppercase
        self.df.iloc[:, 0] = self.df.iloc[:, 0].str.upper()
        
        # Determine max sequence length if not provided
        self.max_length = max(len(seq) for seq in self.df.iloc[:, 0])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sequence = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        
        # One-hot encode the sequence
        encoded_seq = self.one_hot_encode(sequence)
        
        return {
            "x": encoded_seq,
            "y": torch.tensor(label, dtype=torch.float32)
        }
    
    def one_hot_encode(self, seq):
        # Initialize one-hot encoding matrix
        encoding = np.zeros((5, self.max_length), dtype=np.float32)
        
        # Define nucleotide mapping
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        
        # Fill the matrix with one-hot encodings
        for i, nuc in enumerate(seq[:self.max_length]):
            if nuc in nuc_map:
                encoding[nuc_map[nuc], i] = 1.0
            else:
                # Treat any unknown nucleotide as N
                encoding[4, i] = 1.0
        
        return torch.tensor(encoding, dtype=torch.float32)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        
        if args.multiclass == "yes":
            y = y.long()
        else:
            y = y.unsqueeze(1)
        
        optimizer.zero_grad()
        
        outputs = model(x)
        loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y"].numpy()
            
            outputs = model(x)
            
            if args.multiclass == "no":
                probs = outputs.cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
            else:
                probs = outputs.cpu().numpy()
                preds = np.argmax(probs, axis=1)
            
            all_preds.extend(preds)
            all_probs.extend(probs if args.multiclass == "no" else probs.tolist())
            all_labels.extend(y)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    if args.multiclass == "no":
        f1 = f1_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
    else:
        f1 = f1_score(all_labels, all_preds, average="macro")
        mcc = matthews_corrcoef(all_labels, all_preds)
        auc = np.nan
    
    return mcc, auc, f1, accuracy, all_preds, all_probs, all_labels

# Load command line arguments
parser = argparse.ArgumentParser(description='Train simple CNN model for DNA sequence classification.')
parser.add_argument('--data_path', 
                    type=str, 
                    required=True,
                    help='The path of the dataset, specifically, the directory that the train.csv and test.csv lies in')
parser.add_argument('--data_name',
                    type=str,
                    required=True,
                    help='The name of dataset, used for storing results')
parser.add_argument('--multiclass',
                    type=str,
                    choices=["yes", "no"],
                    default="no",
                    help='The number of classes')
parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='Batch size for training')
parser.add_argument('--epochs',
                    type=int,
                    default=50,
                    help='Number of training epochs')
parser.add_argument('--lr',
                    type=float,
                    default=0.0005,
                    help='Learning rate')
args = parser.parse_args()

# Load the data
train_path = f"{args.data_path}/train.csv"
test_path = f"{args.data_path}/test.csv"

train_df = pd.read_csv(train_path, header=0)
test_df = pd.read_csv(test_path, header=0)

# Create datasets and split train into train/validation
train_size = int(0.8 * len(train_df))
val_size = len(train_df) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    train_df, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Convert subsets back to DataFrames
train_subset_df = pd.DataFrame(train_df.iloc[train_subset.indices].values, columns=train_df.columns)
val_subset_df = pd.DataFrame(train_df.iloc[val_subset.indices].values, columns=train_df.columns)

# Create datasets
train_data = SequenceDataset(train_subset_df)
val_data = SequenceDataset(val_subset_df)
test_data = SequenceDataset(test_df)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# Determine number of classes
num_classes = 1 if args.multiclass == "no" else len(train_df.iloc[:, 1].unique())

# Initialize the model
device = torch.device("cpu")
model = SimpleCNN(num_classes=num_classes).to(device)

# Define loss function and optimizer
if args.multiclass == "no":
    criterion = nn.BCELoss()
else:
    criterion = nn.CrossEntropyLoss()

# Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Training loop with validation and early stopping
best_val_loss = float('inf')
patience = 10
patience_counter = 0

print(f"Starting training for {args.data_name} dataset...")
print(f"Multiclass: {args.multiclass}")
print(f"Device: {device}")

for epoch in range(args.epochs):
    
    # Train
    train_loss = train(model, train_loader, criterion, optimizer, device)
    
    # Validate
    model.eval()
    val_loss = 0.0
    val_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            
            if args.multiclass == "yes":
                y = y.long()
            else:
                y = y.unsqueeze(1)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            val_loss += loss.item() * x.size(0)
            val_samples += x.size(0)
    
    val_loss /= val_samples

    
    # Print progress
    print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model weights
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Restore best model
if 'best_model_state' in locals():
    model.load_state_dict(best_model_state)

# Evaluate on test set
mcc, auc, f1, accuracy, all_preds, all_probs, all_labels = evaluate(model, test_loader, device)

print(f"Test Results for {args.data_name}:")
print(f"MCC: {mcc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Record the metrics into a CSV file
results_df = pd.DataFrame({
    'Metric': ['MCC', 'AUC', 'F1-Score', 'Accuracy'],
    'Value': [mcc, auc, f1, accuracy]
})

# Save results
results_dir = f"{project_dir}/results_final/baseline"
preds_dir = f"{project_dir}/preds/baseline"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(preds_dir, exist_ok=True)

# Save results
results_df.to_csv(f"{results_dir}/{args.data_name}_rf.csv", index=False)

# Save predictions
if args.multiclass == "no":
    preds_table = pd.DataFrame({
        'True': all_labels,
        'Pred': all_probs
    })
    preds_table.to_csv(f"{preds_dir}/{args.data_name}_rf.csv", index=False)