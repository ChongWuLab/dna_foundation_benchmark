import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import random
import time
import os


random.seed(42)

checkpoint = "Path/to/the/downloaded/model/checkpoint"
device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(checkpoint, 
                                          trust_remote_code=True,
                                          local_files_only=True)
model = AutoModelForMaskedLM.from_pretrained(checkpoint, 
                                  trust_remote_code=True,
                                  local_files_only=True)
model.eval()
model.to(device)

# Function to generate random DNA sequences
def generate_random_dna_sequences(n_sequences, seq_length):
    
    dna_bases = ['A', 'T', 'C', 'G']
    sequences = []
    
    for _ in range(n_sequences):
        # Generate a random sequence of the specified length
        sequence = ''.join(random.choice(dna_bases) for _ in range(seq_length))
        sequences.append(sequence)
    
    return sequences

# Function to tokenize and run inference
def run_inference(sequences, tokenizer, model, device, max_length):
    
    try:
        # Tokenize the sequences with adjusted max_length
        inputs = tokenizer(
            sequences, 
            padding='max_length', 
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
        
        # Run forward pass
        with torch.no_grad():
            start_time = time.time()
            outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
            end_time = time.time()
            
        torch.cuda.empty_cache()
        runtime = end_time - start_time
        return runtime
    
    except Exception as e:
        print("Exceeding maximum capacity of model")
        print(f"Error: {e}")
        return None

# Fixed number of sequences per batch and repetitions
n_sequences = 1
n_repetitions = 100

# Define sequence lengths
seq_lengths = [2**i for i in range(5, 14)]  # 32, ..., 524288
results = {}

# Run experiments for each sequence length
for seq_length in seq_lengths:
    # Set max_length by definition of NT-v2
    max_length = (seq_length // 6) + (seq_length % 6)
    
    print(f"\nRunning experiment with sequence length {seq_length}, max_length for tokenizer: {max_length}")
    
    # Generate sequences once for this length - will be reused in all repetitions
    sequences = generate_random_dna_sequences(
        n_sequences=n_sequences, 
        seq_length=seq_length
    )
    
    total_runtime = 0
    successful_runs = 0
    
    for i in range(n_repetitions):

        # Run inference and measure runtime using the same sequences
        runtime = run_inference(
            sequences, 
            tokenizer, 
            model, 
            device, 
            max_length=max_length
        )
        
        if runtime is None:
            print(f"Error occurred at repetition {i+1} for sequence length {seq_length}")
            break
        
        total_runtime += runtime
        successful_runs += 1
        
        if (i+1) % 100 == 0:
            print(f"Completed {i+1}/{n_repetitions} repetitions for sequence length {seq_length}")

    
    # Save results if there were any successful runs
    if successful_runs > 0:
        results[seq_length] = {
            'total_runtime': total_runtime,
            'successful_runs': successful_runs,
            'n_sequences': n_sequences,
        }
        print(f"Successfully completed {successful_runs} runs for sequence length {seq_length}")
        print(f"Total runtime: {total_runtime:.4f} seconds")
    else:
        print(f"No successful runs for sequence length {seq_length}")

output_path = "../runtimes/ntv2.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

rows = []
for seq_length, data in results.items():
    rows.append({
        'sequence_length': seq_length,
        'n_sequences': n_sequences,
        'successful_runs': data['successful_runs'],
        'total_runtime_seconds': data['total_runtime'],
    })

df = pd.DataFrame(rows)
df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
print("All experiments completed!")