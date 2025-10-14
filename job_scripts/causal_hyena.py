import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import os
import sys
import gc


project_dir = ".."
checkpoint = "Path/to/the/downloaded/model/checkpoint"
INPUT_SEQUENCE_LENGTH = 6000
MAX_LENGTH = INPUT_SEQUENCE_LENGTH + 1
BATCH_SIZE = 128

# Helper functions

def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """Mean pool hidden states using an attention mask."""
    attention_mask = attention_mask.unsqueeze(-1)
    embed = torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
    return embed

def _embed_batch_hyena(seq_list, tokenizer, model, max_length):
    """Internal function to embed a single batch with the HyenaDNA model."""
    tok = tokenizer(
        seq_list,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    tok = {k: v.to(model.device) for k, v in tok.items()}
    
    with torch.no_grad():
        attn_mask = (tok["input_ids"] != tokenizer.pad_token_id).int()
        hidden_states = model(tok["input_ids"])[0]
    
    embedding = mean_pooling(hidden_states, attn_mask)
    return embedding

def get_hyena_embeddings(model, tokenizer, sequences, batch_size, max_length):
    """Processes a list of sequences in batches to get HyenaDNA embeddings."""
    all_embeddings_np = []
    num_sequences = len(sequences)
    num_batches = -(num_sequences // -batch_size)

    for i in range(0, num_sequences, batch_size):
        batch_seqs = sequences[i : i + batch_size]
        batch_embeddings = _embed_batch_hyena(batch_seqs, tokenizer, model, max_length)
        all_embeddings_np.append(batch_embeddings.cpu().numpy())

        del batch_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Processing batch {(i // batch_size) + 1} / {num_batches}...")

    embeddings_matrix = np.concatenate(all_embeddings_np, axis=0)
    return embeddings_matrix

def calculate_and_save_metrics(vectors1, vectors2, results_df, id_columns, output_dir, prefix, file_type):
    """Calculates metrics, combines them with identifiers, and saves to files."""
    print(f"Calculating and saving metrics for prefix: {prefix}, type: {file_type}")
    os.makedirs(output_dir, exist_ok=True)

    differences = vectors2 - vectors1
    results_df[f'{prefix}_L1'] = np.sum(np.abs(differences), axis=1)
    results_df[f'{prefix}_L2'] = np.linalg.norm(differences, axis=1)
    
    norm1 = np.linalg.norm(vectors1, axis=1)
    norm2 = np.linalg.norm(vectors2, axis=1)
    dot_product = np.sum(vectors1 * vectors2, axis=1)
    cosine_sim = np.divide(dot_product, norm1 * norm2,
                           out=np.zeros_like(dot_product, dtype=float),
                           where=(norm1 * norm2) != 0)
    results_df[f'{prefix}_cosine_similarity'] = cosine_sim

    # Save the combined metrics file
    metrics_output_path = f"{output_dir}/{prefix}_metrics_{file_type}.csv"
    results_df.to_csv(metrics_output_path, index=False)
    print(f"Saved combined metrics to {metrics_output_path}")

    # Save the difference vectors with identifiers
    diff_df = pd.DataFrame(differences)
    diff_df.columns = [f'diff_feature_{i}' for i in range(diff_df.shape[1])]
    full_diff_df = pd.concat([results_df[id_columns].reset_index(drop=True), diff_df], axis=1)
    diff_output_path = f"{output_dir}/{prefix}_differences_{file_type}.csv"
    full_diff_df.to_csv(diff_output_path, index=False)
    print(f"Saved difference vectors to {diff_output_path}")






tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True, device_map="auto")
model.eval()
print(f"Model loaded successfully. Main device: {model.device}")

folders = ['ipaqtl', 'paqtl', 'sqtl', 'eqtl']
file_types = ['neg', 'pos']

for folder in folders:
    for file_type in file_types:
        input_file_path = f"{project_dir}/data_processed/causal/{folder}/{file_type}_seqs_medium.csv"
        output_dir = f"{project_dir}/data_processed/causal/{folder}"

        print(f"\nProcessing file: {input_file_path}")

        df = pd.read_csv(input_file_path)
        id_columns_to_keep = ['chromosome', 'pos', 'ref', 'alt']
        try:
            results_df = df[id_columns_to_keep].copy()
            print(f"Found and stored identifier columns: {id_columns_to_keep}")
        except KeyError:
            print(f"WARNING: Identifier columns not found. Proceeding without them.")
            results_df = pd.DataFrame(index=df.index)
            id_columns_to_keep = []

        sequences1 = df.iloc[:, 0].astype(str).tolist()
        sequences2 = df.iloc[:, 1].astype(str).tolist()
        num_seqs = len(sequences1)

        print(f"Generating embeddings for {num_seqs} sequences in Seq1...")
        embeddings1 = get_hyena_embeddings(model, tokenizer, sequences1, BATCH_SIZE, MAX_LENGTH)

        print(f"Generating embeddings for {num_seqs} sequences in Seq2...")
        embeddings2 = get_hyena_embeddings(model, tokenizer, sequences2, BATCH_SIZE, MAX_LENGTH)

        print(f"Embeddings generated. Shape: {embeddings1.shape}")

        calculate_and_save_metrics(
            vectors1=embeddings1,
            vectors2=embeddings2,
            results_df=results_df,
            id_columns=id_columns_to_keep,
            output_dir=output_dir,
            prefix="hyena",
            file_type=file_type
        )

        print(f"Successfully processed and saved results for {input_file_path}")

print("\nAll processing complete!")