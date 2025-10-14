import os
import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer

###############################################################################
# 1. FASTA Parsing
###############################################################################
def parse_fasta(fasta_path):
    """
    Returns a list of (header, sequence) from a FASTA file.
    Sequences are uppercased for consistency.
    """
    records = []
    header = None
    seq_chunks = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # store previous
                if header and seq_chunks:
                    full_seq = "".join(seq_chunks).upper()
                    records.append((header, full_seq))
                header = line[1:].strip()  # remove '>'
                seq_chunks = []
            else:
                seq_chunks.append(line)
        # last
        if header and seq_chunks:
            full_seq = "".join(seq_chunks).upper()
            records.append((header, full_seq))

    return records

###############################################################################
# 2. Main
###############################################################################
def main():
    # Config
    project_dir = ".."
    fasta_path = f"{project_dir}/data_processed/TAD/background_6kb_sequences.fa"
    checkpoint = "Path/to/the/downloaded/model/checkpoint"
    max_length = 1000
    device     = torch.device("cuda")

    # Load FASTA
    seq_records = parse_fasta(fasta_path)  

    # Model + Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,
                                              trust_remote_code=True,
                                              local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint,
                                                 trust_remote_code=True,
                                                 local_files_only=True)
    model.to(device)
    model.eval()

    # Initialize global sum => shape [max_length, max_length]
    # We'll store float32 on CPU
    global_sum = np.zeros((max_length, max_length), dtype=np.float32)
    total_count = len(seq_records)

    # Processing loop
    for idx, (header, seq) in enumerate(seq_records):
        # 1) Tokenize
        # If sequence is >6000, we truncate. If <6000, we pad with model's mechanism.
        tok = tokenizer(seq, 
                        padding='max_length',
                        truncation=True,
                        max_length=max_length,
                        return_tensors='pt')
        tok = {k: v.to(device) for k, v in tok.items()}

        # 2) Forward => get attentions
        with torch.no_grad():
            out = model(tok["input_ids"], 
                         attention_mask=tok["attention_mask"],
                         output_attentions=True)
        
        # out.attentions => tuple of length = num_layers (29 for NTv2)
        # each shape => [batch_size=1, num_heads=16, seq_len, seq_len]
        # We'll average across layers (dim=0) & heads (dim=1).
        # => final shape [1, seq_len, seq_len], then remove batch dim => [seq_len, seq_len]
        # Then convert to CPU + np array.

        # 'attentions' is a tuple of length 29
        # so we stack them => shape [29, 1, 16, seq_len, seq_len]
        # but we only have batch_size=1 => so final shape is [29, 16, seq_len, seq_len]
        attentions = out.attentions  
        layer_list = []
        for a in attentions:
            layer_list.append(a[0].detach().cpu().numpy())  
            # shape of a => [1,16,seq_len,seq_len], remove batch => [16, seq_len, seq_len]
        # stack => shape [29, 16, seq_len, seq_len]
        attn_4d = np.stack(layer_list, axis=0)

        # average over layers & heads => shape [seq_len, seq_len]
        # axis=0 => layers, axis=1 => heads => total of 2 dimensions
        attn_mean_2d = attn_4d.mean(axis=(0,1))

        # add to global sum => must match float32
        global_sum += attn_mean_2d.astype(np.float32)

        # cleanup
        del attentions, layer_list, attn_4d, attn_mean_2d, out
        torch.cuda.empty_cache()

        if (idx+1) % 100 == 0:
            print(f"Processed {idx+1}/{total_count} sequences...")

    global_avg = global_sum / float(total_count)

    # Save final average
    os.makedirs(f"{project_dir}/attention_weights", exist_ok=True)
    np.save(f"{project_dir}/attention_weights/ntv2_background.npy", global_avg)
    print(f"Saved final average attention => shape={global_avg.shape}")

if __name__ == "__main__":
    main()
