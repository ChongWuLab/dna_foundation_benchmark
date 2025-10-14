import torch
import numpy as np
import pandas as pd
from enformer_pytorch import from_pretrained, seq_indices_to_one_hot
import os
import sys
import gc


project_dir = ".."
checkpoint = "Path/to/the/downloaded/model/checkpoint"
INPUT_SEQUENCE_LENGTH = 196608
BATCH_SIZE = 8



def get_center_sequence(full_sequence: str, target_length: int):
    """Extracts the central part of a sequence."""
    start = (len(full_sequence) - target_length) // 2
    end = start + target_length
    return full_sequence[start:end]

def seq_to_indices(seq: str) -> np.ndarray:
    """Converts a DNA sequence string to a numpy array of indices."""
    seq_bytes = seq.upper().encode("ascii")
    arr = np.frombuffer(seq_bytes, dtype=np.uint8)
    arr = arr - 65
    lut = np.full(26, 4, dtype=np.int8)
    for base, idx in zip("ACGT", range(4)):
        lut[ord(base) - 65] = idx
    out = np.full(arr.shape, 4, dtype=np.int8)
    valid = (arr >= 0) & (arr < 26)
    out[valid] = lut[arr[valid]]
    return out

def one_hot_encode_sequences(sequences: list[str], device: torch.device):
    """Converts a list of DNA strings to a one-hot encoded tensor."""
    indices_list = [seq_to_indices(s) for s in sequences]
    indices_tensor = torch.from_numpy(np.stack(indices_list)).long()
    one_hot_tensor = seq_indices_to_one_hot(indices_tensor).to(device)
    return one_hot_tensor

def get_enformer_vectors(model, sequences, batch_size, device):
    """
    Processes sequences in batches to get both hidden states and final outputs from Enformer.
    """
    all_hidden_states_np = []
    all_outputs_np = []
    num_sequences = len(sequences)
    num_batches = -(num_sequences // -batch_size)

    with torch.no_grad():
        for i in range(0, num_sequences, batch_size):
            batch_seqs = sequences[i : i + batch_size]
            input_tensor = one_hot_encode_sequences(batch_seqs, device)
            output, hidden = model(input_tensor, return_embeddings=True)

            hidden_vec = hidden.mean(dim=1)
            all_hidden_states_np.append(hidden_vec.cpu().numpy())
            output_vec = output['human'].mean(dim=1)
            all_outputs_np.append(output_vec.cpu().numpy())

            del input_tensor, output, hidden, hidden_vec, output_vec
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f"Processing batch {(i // batch_size) + 1} / {num_batches}...")

    hidden_states_matrix = np.concatenate(all_hidden_states_np, axis=0)
    outputs_matrix = np.concatenate(all_outputs_np, axis=0)
    return hidden_states_matrix, outputs_matrix

def calculate_and_save_metrics(vectors1, vectors2, base_results_df, id_columns, output_dir, prefix, file_type):
    """Calculates metrics and saves them to separate files for a given prefix."""
    print(f"Calculating and saving metrics for prefix: {prefix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a fresh copy of the results DataFrame for this specific output type
    results_df = base_results_df.copy()

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





print("Loading Enformer model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = from_pretrained(checkpoint, local_files_only=True)
model.to(device)
model.eval()
print(f"Model loaded successfully on {device}.")

folders = ['ipaqtl', 'paqtl', 'sqtl', 'eqtl']
file_types = ['neg', 'pos']

for folder in folders:
    for file_type in file_types:
        input_file_path = f"{project_dir}/data_processed/causal/{folder}/{file_type}_seqs_long.csv"
        output_dir = f"{project_dir}/data_processed/causal/{folder}"

        print(f"\nProcessing file: {input_file_path}")

        df = pd.read_csv(input_file_path)
        id_columns_to_keep = ['chromosome', 'pos', 'ref', 'alt']
        try:
            base_results_df = df[id_columns_to_keep].copy()
            print(f"Found and stored identifier columns: {id_columns_to_keep}")
        except KeyError:
            print(f"WARNING: Identifier columns not found. Proceeding without them.")
            base_results_df = pd.DataFrame(index=df.index)
            id_columns_to_keep = []

        raw_sequences1 = df.iloc[:, 0].astype(str).tolist()
        raw_sequences2 = df.iloc[:, 1].astype(str).tolist()

        print(f"Cropping {len(raw_sequences1)} sequence pairs to center {INPUT_SEQUENCE_LENGTH} bases...")
        cropped_seqs1 = [get_center_sequence(s, INPUT_SEQUENCE_LENGTH) for s in raw_sequences1]
        cropped_seqs2 = [get_center_sequence(s, INPUT_SEQUENCE_LENGTH) for s in raw_sequences2]
        num_seqs = len(cropped_seqs1)

        print(f"Generating embeddings for {num_seqs} sequences in Seq1...")
        hidden1, output1 = get_enformer_vectors(model, cropped_seqs1, BATCH_SIZE, device)

        print(f"Generating embeddings for {num_seqs} sequences in Seq2...")
        hidden2, output2 = get_enformer_vectors(model, cropped_seqs2, BATCH_SIZE, device)

        print(f"Embeddings generated. Shapes (hidden/output): {hidden1.shape} / {output1.shape}")

        # Calculate and save metrics for hidden states
        calculate_and_save_metrics(hidden1, hidden2, base_results_df, id_columns_to_keep, output_dir, "enformer_hidden", file_type)

        # Calculate and save metrics for output tracks
        calculate_and_save_metrics(output1, output2, base_results_df, id_columns_to_keep, output_dir, "enformer_output", file_type)

        print(f"Successfully processed and saved results for {input_file_path}")

print("\nAll processing complete!")