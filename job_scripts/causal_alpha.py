import numpy as np
import pandas as pd
from alphagenome.models import dna_client
from alphagenome.models.dna_client import OutputType
import os
import time
import grpc


INPUT_SEQUENCE_LENGTH = 131072
CENTER_WINDOW = 2048
project_dir = ".."
api_key = "Your API Key"


def get_center_sequence(full_sequence: str, target_length: int):
    """Extracts the central part of a sequence."""
    start = (len(full_sequence) - target_length) // 2
    end = start + target_length
    return full_sequence[start:end]

def get_centered_feature_vector(prediction_result, center_window_bp=2048):
    output_attrs = [
        'atac', 'cage', 'dnase', 'rna_seq', 'chip_histone', 'chip_tf',
        'splice_sites', 'splice_site_usage', 'procap'
    ]
    feature_vectors = []
    for attr_name in output_attrs:
        track_data = getattr(prediction_result, attr_name)
        track_values = track_data.values
        num_bins = track_values.shape[0]
        resolution = INPUT_SEQUENCE_LENGTH / num_bins
        center_bin = num_bins // 2
        window_bins = int(center_window_bp / resolution)
        half_window_bins = window_bins // 2
        start_bin = center_bin - half_window_bins
        end_bin = center_bin + half_window_bins
        central_window_predictions = track_values[start_bin:end_bin, :]
        averaged_vector = np.mean(central_window_predictions, axis=0)
        feature_vectors.append(averaged_vector)
    return np.concatenate(feature_vectors)



# Initialize the model client
try:
    dna_model = dna_client.create(api_key)
    print("AlphaGenome client created successfully.")
except Exception as e:
    print(f"Error creating AlphaGenome client: {e}")
    exit()

# Define folders and file structure
folders = ['ipaqtl', 'paqtl', 'sqtl']
file_types = ['neg', 'pos']

print("Starting File-by-File, Sequence-by-Sequence Processing:")

for folder in folders:
    for file_type in file_types:
        input_file_path = f"{project_dir}/data_processed/causal/{folder}/{file_type}_seqs_long.csv"
        print(f"\nProcessing file: {input_file_path}")
        
        try:
            df = pd.read_csv(input_file_path)
            id_columns_to_keep = ['chromosome', 'pos', 'ref', 'alt']
            results_df = df[id_columns_to_keep].copy()
            print(f"Found and stored identifier columns: {id_columns_to_keep}")
        except KeyError:
            print(f"WARNING: Could not find all expected ID columns {id_columns_to_keep} in the input file.")
            print("Proceeding without identifier columns.")
            results_df = pd.DataFrame()


        sequences1 = [get_center_sequence(s, INPUT_SEQUENCE_LENGTH) for s in df.iloc[:, 0].tolist()]
        sequences2 = [get_center_sequence(s, INPUT_SEQUENCE_LENGTH) for s in df.iloc[:, 1].tolist()]
        
        num_pairs = len(sequences1)
        
        vectors1 = []
        vectors2 = []

        print(f"Predicting for {num_pairs} sequences in Seq1...")
        for i in range(num_pairs):
            if (i + 1) % 10 == 0:
                print(f"Processing Seq1, sequence {i+1}/{num_pairs}...")

            while True:
                try:
                    result = dna_model.predict_sequence(
                        sequence=sequences1[i],
                        requested_outputs=[output for output in OutputType],
                        ontology_terms=None
                    )
                    vectors1.append(get_centered_feature_vector(result, CENTER_WINDOW))
                    break
                except grpc.RpcError as e:
                    if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                        print(f"\nQuota exceeded on sequence {i+1}. Waiting 60s...")
                        time.sleep(60)
                    else:
                        raise e
        
        print(f"Predicting for {num_pairs} sequences in Seq2...")
        for i in range(num_pairs):
            if (i + 1) % 10 == 0:
                print(f"Processing Seq2, sequence {i+1}/{num_pairs}...")
                
            while True:
                try:
                    result = dna_model.predict_sequence(
                        sequence=sequences2[i],
                        requested_outputs=[output for output in OutputType],
                        ontology_terms=None
                    )
                    vectors2.append(get_centered_feature_vector(result, CENTER_WINDOW))
                    break
                except grpc.RpcError as e:
                    if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                        print(f"\nQuota exceeded on sequence {i+1}. Waiting 60s...")
                        time.sleep(60)
                    else:
                        raise e

        vectors1_matrix = np.array(vectors1)
        vectors2_matrix = np.array(vectors2)

        # Calculate metrics and add them as new columns to our results_df
        differences = vectors2_matrix - vectors1_matrix
        
        # Add calculated metrics to the results DataFrame
        results_df['alpha_L1'] = np.sum(np.abs(differences), axis=1)
        results_df['alpha_L2'] = np.linalg.norm(differences, axis=1)
        
        norm1 = np.linalg.norm(vectors1_matrix, axis=1)
        norm2 = np.linalg.norm(vectors2_matrix, axis=1)
        dot_product = np.sum(vectors1_matrix * vectors2_matrix, axis=1)
        cosine_similarities = np.divide(dot_product, norm1 * norm2, 
                                        out=np.zeros_like(dot_product, dtype=float), 
                                        where=(norm1*norm2)!=0)
        results_df['alpha_cosine_similarity'] = cosine_similarities

        # Save the combined metrics file
        output_metrics_path = f"{project_dir}/data_processed/causal/{folder}/alpha_metrics_{file_type}.csv"
        results_df.to_csv(output_metrics_path, index=False)
        print(f"Successfully saved combined metrics to {output_metrics_path}")

        # Save the raw difference vectors with identifiers as well
        diff_df = pd.DataFrame(differences)
        diff_df.columns = [f'diff_feature_{i}' for i in range(diff_df.shape[1])]
        # Concatenate with the ID columns
        full_diff_df = pd.concat([results_df[id_columns_to_keep].reset_index(drop=True), diff_df], axis=1)
        
        output_diff_path = f"{project_dir}/data_processed/causal/{folder}/alpha_differences_{file_type}.csv"
        full_diff_df.to_csv(output_diff_path, index=False)
        print(f"Successfully saved difference vectors to {output_diff_path}")
            
print("\nAll files processed successfully!")