import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
import sys
import gc


project_dir = ".."
checkpoint = "Path/to/the/downloaded/model/checkpoint"
INPUT_SEQUENCE_LENGTH = 6000
MAX_LENGTH = INPUT_SEQUENCE_LENGTH + 1
BATCH_SIZE = 128


# Helper functions

def reverse_complement(sequence: str):
    """Reverse complement a DNA sequence."""
    complement_map = str.maketrans("ACGTN", "TGCAN")
    return sequence.upper().translate(complement_map)[::-1]

def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """Mean pool hidden states using an attention mask."""
    attention_mask = attention_mask.unsqueeze(-1)
    embed = torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
    return embed

def _embed_batch_caduceus(seq_list, tokenizer, model, max_length):
    """
    Internal function to embed a single batch with Caduceus, using
    forward and reverse-complement averaging.
    """
    tok_fwd = tokenizer(seq_list, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    rc_list = [reverse_complement(seq) for seq in seq_list]
    tok_rc = tokenizer(rc_list, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    
    device = model.device
    tok_fwd = {k: v.to(device) for k, v in tok_fwd.items()}
    tok_rc  = {k: v.to(device) for k, v in tok_rc.items()}
    
    with torch.no_grad():
        out_fwd = model(tok_fwd["input_ids"], output_hidden_states=True).hidden_states[-1]
        out_rc  = model(tok_rc["input_ids"],  output_hidden_states=True).hidden_states[-1]
        
    hidden_avg = (out_fwd + out_rc) / 2.0
    attn_mask = (tok_fwd["input_ids"] != tokenizer.pad_token_id).int()
    embedding = mean_pooling(hidden_avg, attn_mask)
    return embedding

def get_caduceus_embeddings(model, tokenizer, sequences, batch_size, max_length):
    """
    Processes a list of sequences in batches to get Caduceus embeddings.
    """
    all_embeddings_np = []
    num_sequences = len(sequences)
    num_batches = -(num_sequences // -batch_size)

    for i in range(0, num_sequences, batch_size):
        batch_seqs = sequences[i : i + batch_size]
        
        print(f"Processing batch {(i // batch_size) + 1} / {num_batches}...")

        batch_embeddings = _embed_batch_caduceus(batch_seqs, tokenizer, model, max_length)
        all_embeddings_np.append(batch_embeddings.cpu().numpy())

        del batch_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    embeddings_matrix = np.concatenate(all_embeddings_np, axis=0)
    return embeddings_matrix



def calculate_and_save_metrics(vectors1, vectors2, results_df, id_columns, output_dir, prefix, file_type):
    """
    Calculates metrics, combines them with identifiers from results_df, and saves to files.
    """
    print(f"Calculating and saving metrics for prefix: {prefix}, type: {file_type}")
    os.makedirs(output_dir, exist_ok=True)

    differences = vectors2 - vectors1
    l1 = np.sum(np.abs(differences), axis=1)
    l2 = np.linalg.norm(differences, axis=1)
    norm1 = np.linalg.norm(vectors1, axis=1)
    norm2 = np.linalg.norm(vectors2, axis=1)
    dot_product = np.sum(vectors1 * vectors2, axis=1)
    cosine_sim = np.divide(dot_product, norm1 * norm2,
                           out=np.zeros_like(dot_product, dtype=float),
                           where=(norm1 * norm2) != 0)

    # Add calculated metrics as new columns to the results DataFrame
    results_df[f'{prefix}_L1'] = l1
    results_df[f'{prefix}_L2'] = l2
    results_df[f'{prefix}_cosine_similarity'] = cosine_sim

    # Save the combined metrics file
    metrics_output_path = f"{output_dir}/{prefix}_metrics_{file_type}.csv"
    results_df.to_csv(metrics_output_path, index=False)
    print(f"Saved combined metrics to {metrics_output_path}")

    # Save the difference vectors with identifiers
    diff_df = pd.DataFrame(differences)
    diff_df.columns = [f'diff_feature_{i}' for i in range(diff_df.shape[1])]
    
    # Concatenate ID columns with the difference vectors
    full_diff_df = pd.concat([results_df[id_columns].reset_index(drop=True), diff_df], axis=1)
    
    diff_output_path = f"{output_dir}/{prefix}_differences_{file_type}.csv"
    full_diff_df.to_csv(diff_output_path, index=False)
    print(f"Saved difference vectors to {diff_output_path}")




print("Loading Caduceus-Ph tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
model = AutoModelForMaskedLM.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True, device_map="auto")
model.eval()
print(f"Model loaded successfully. Main device: {model.device}")

folders = ['ipaqtl', 'paqtl', 'sqtl', 'eqtl']
file_types = ['neg', 'pos']

for folder in folders:
    for file_type in file_types:
        input_file_path = f"{project_dir}/data_processed/causal/{folder}/{file_type}_seqs_medium.csv"
        output_dir = f"{project_dir}/data_processed/causal/{folder}"

        print(f"\nProcessing file: {input_file_path}")
        
        # Read input and store identifier columns.
        df = pd.read_csv(input_file_path)
        id_columns_to_keep = ['chromosome', 'pos', 'ref', 'alt']
        try:
            results_df = df[id_columns_to_keep].copy()
            print(f"Found and stored identifier columns: {id_columns_to_keep}")
        except KeyError:
            print(f"WARNING: Could not find all expected ID columns {id_columns_to_keep} in {input_file_path}.")
            print("Proceeding without identifier columns.")
            results_df = pd.DataFrame(index=df.index)
            id_columns_to_keep = []


        sequences1 = df.iloc[:, 0].astype(str).tolist()
        sequences2 = df.iloc[:, 1].astype(str).tolist()

        num_seqs = len(sequences1)

        print(f"Generating embeddings for {num_seqs} sequences in Seq1...")
        embeddings1 = get_caduceus_embeddings(model, tokenizer, sequences1, BATCH_SIZE, MAX_LENGTH)

        print(f"Generating embeddings for {num_seqs} sequences in Seq2...")
        embeddings2 = get_caduceus_embeddings(model, tokenizer, sequences2, BATCH_SIZE, MAX_LENGTH)

        print(f"Embeddings generated. Shape: {embeddings1.shape}")


        calculate_and_save_metrics(
            vectors1=embeddings1,
            vectors2=embeddings2,
            results_df=results_df,
            id_columns=id_columns_to_keep,
            output_dir=output_dir,
            prefix="cadph",
            file_type=file_type
        )

        print(f"Successfully processed and saved results for {input_file_path}")

print("\nAll processing complete!")