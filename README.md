# DNA Foundation Models Benchmarking
## Introduction
This repository is for generating the results of DNA foundation models benchmarking.

Please cite the following manuscript for using DNAm models built and association results by our work:  


> Feng, Haonan, Lang Wu, Bingxin Zhao, Chad Huff, Jianjun Zhang, Jia Wu, Lifeng Lin, Peng Wei, and Chong Wu. "[Benchmarking DNA Foundation Models for Genomic and Genetic Tasks]([https://www.biorxiv.org/content/10.1101/2024.08.16.608288v1)." bioRxiv (2024): 2024-08.

<p align="center">
  <img src="https://github.com/ChongWuLab/dna_foundation_benchmark/blob/main/Fig1.png" width=50% height=50%>
</p>

## 1. Environment Setup
To ensure reproducibility and avoid dependency conflicts, we **strongly recommend** using separate virtual environments (e.g., `virtualenv`) or Docker images for each DNA foundation model you intend to use. This mirrors our own development and testing process.

*   **Python Environment:** Please refer to the official GitHub repository of each specific DNA foundation model for detailed instructions on setting up their required Python environment and installing dependencies. Each model may have unique requirements.
*   **Hardware:** **GPU is required** for running these DNA foundation models and our associated code efficiently. All our experiments were conducted on GPU-accelerated hardware.
*   **DNA Foundation Model Loading** In our workflow, all DNA foundation models were downloaded from Huggingface and were loaded using `from_pretrained()` with the argument `checkpoint` specifying the local model path, and the argument `local_files_only=True`. Feel free to change the loading process. 

## 2. Sequence Classification Benchmark

This section outlines the steps to generate results for the sequence classification benchmark tasks.
### Datasets

We curated and preprocessed datasets for various genomic tasks from published works. These datasets consist of DNA sequences and their corresponding labels. We further processed them to create standardized train-test splits.

The processed datasets can be downloaded from Huggingface: https://huggingface.co/datasets/hfeng3/dna_foundation_benchmark_dataset. After downloading, unzip and put all files inside the data_processed directory. After downloading and extracting, each individual dataset directory will contain `train.csv` and `test.csv` files. The resultant directory structure should be:
```
dna_foundation_benchmark/
│
├── data_processed/
│   ├── enhancers/
│   │   ├── enhancer/
│   │   │   ├──train.csv
│   │   │   ├──test.csv
│   │   ├── enhancer_strength/
│   │   │   ├──train.csv
│   │   │   ├──test.csv
│   │── ...   
├── analysis/
├── job_scripts/
```
The path to the directory (containing `train.csv` and `test.csv`) will be used as the `--data_path` argument in the inference scripts.

### Generating Zero-Shot Embeddings

To generate zero-shot embeddings for a given dataset using a specific DNA foundation model:

1.  **Navigate to the scripts directory:**
    ```bash
    cd job_scripts
    ```

2.  **Run the inference script:**
    Execute the Python script corresponding to the DNA foundation model you wish to use (e.g., `inference_cadph.py` for Caduceus-Ph, `inference_dnabert.py` for DNABERT, etc.).

    **Arguments:**
    *   `--data_path`: Path to the specific dataset folder (e.g., `../data_processed/iPro-WAEL/Promoter_Arabidopsis_NonTATA`). Adjust this path based on where you downloaded and extracted the datasets relative to the `jobscripts` directory.
    *   `--data_name`: The name of the dataset folder (e.g., `Promoter_Arabidopsis_NonTATA`). This is often the same as the last component of `--data_path`.
    *   `--max_length`: The maximum sequence length to use after tokenization. Sequences longer than this will be truncated.
    *   `--pooling`: The pooling method to apply to the model's output embeddings. Common options include `mean`, `max`, or `cls`.

    **Example:**
    To generate zero-shot embeddings using the **Caduceus-Ph** model for the **Promoter_Arabidopsis_NonTATA** dataset, with a maximum tokenized length of **700** and using **max pooling**, run the following command from within the `job_scripts` directory:

    ```bash
    python inference_cadph.py \
        --data_path ../data_processed/iPro-WAEL/Promoter_Arabidopsis_NonTATA \
        --data_name Promoter_Arabidopsis_NonTATA \
        --max_length 700 \
        --pooling max
    ```
    *(Note: The `../data_processed/` part of the `--data_path` assumes your datasets are stored in a `data_processed` directory one level above the `job_scripts` directory. Please adjust this path according to your local file structure.)*

The generated embeddings will by default be saved to `embeddings/[DATASET_NAME]` as csv files. For the example above, the output would be in `embeddings/Promoter_Arabidopsis_NonTATA`.

### Classification on Embeddings

Once the zero-shot embeddings have been generated, you can train simple classifiers on them to get the final performance metrics.

1.  **Stay in the `jobscripts` directory.**

2.  **Run the classification script:**
    Run the `classify_[model_short_name].py` script corresponding to the model whose embeddings you want to evaluate. The script loads the embeddings you generated in the previous step and uses them to train a classifier.

    **Arguments:**
    *   `--data_name`: Name of the dataset.
    *   `--pooling`: The pooling method used during the embedding generation step (e.g., `max`, `mean`). This is crucial for loading the correct embedding file.
    *   `--multiclass`: Specify `yes` or `no` to indicate if the dataset is for a multi-class task. This is important for calculating the correct performance metrics.
    *   `--classifier`: The name of the simple classifier to train. Options include `random_forest`, `naive_bayes`, or `elastic` (for Elastic Net logistic regression).

    **Example:**
    To train an **Elastic Net** classifier on the **Caduceus-Ph** embeddings for the **Promoter_Arabidopsis_NonTATA** dataset (which were generated with **max pooling** and is a binary classification task), run the following command:

    ```bash
    python classify_cadph.py \
        --data_name Promoter_Arabidopsis_NonTATA \
        --pooling max \
        --multiclass no \
        --classifier elastic
    ```

    **Output:**
    The script will train the classifier, evaluate its performance, and save a summary of the metrics. The output for this example will be generated in `results_final/cadph/`.

### Result Analysis and Comparison

To organize the generated metrics and replicate the comparison tables from our work, navigate to the `analysis` directory:
```bash
cd analysis
```
From here, you can run the following scripts to aggregate results:

*   `compare_across_classifier.py`: Compares different classifiers for the same model and pooling method.
*   `compare_across_model.py`: Compares the performance of different DNA foundation models.
*   `compare_across_pooling.py`: Compares the impact of different pooling methods.
*   `compare_pretrain_vs_1k.py`: A specific comparison for HyenaDNA checkpoints. This assumes you have already pretrained a HyenaDNA model, obtained its classification results in the same format as our other experiments, and have also run `inference_hyena_1k.py` and `classify_hyena_1k.py` from the `job_scripts` directory.

To replicate the boxplots （Figure 2 and Supp Fig 1), run boxplot_classifier.py and boxplot_pooling.py.

### Statistical Significance (DeLong's Test)

To statistically compare the Area Under the ROC Curve (AUC) between different results, we provide scripts that implement DeLong's test. From the `analysis` directory, run the appropriate script to get a list of winners with a significance of p < 0.01.

*   `delong_across_models.py`: Compares AUCs between different models.
*   `delong_across_pooling.py`: Compares AUCs between different pooling methods.
*   `delong_pretrain_vs_1k.py`: Compares AUCs for the HyenaDNA checkpoint evaluation.

### Comparison with CNN Baseline

To compare the performance of foundation models against a baseline CNN model:

1.  From the `job_scripts` directory, run `classify_baseline.py` for each dataset.
2.  Ensure you have already generated results for the foundation models using `mean` pooling.
3.  Navigate to the `analysis` directory.
4.  Run `compare_meanpool_vs_baseline.py` to aggregate the metrics and `delong_meanpool_vs_baseline.py` to perform the statistical comparison.


## 3. Gene Expression Prediction Benchmark

**Please note:** The code for this section was adapted for a specific high-performance computing (HPC) server environment. As such, it contains server-specific arguments and code structures. This section is provided as a general reference for our methodology rather than a direct, universally runnable guide.

### Data Acquisition

*   **DNA Sequences:** Access to the required individual DNA sequences is protected and must be applied for through the GTEx Portal: [https://gtexportal.org/home/protectedDataAccess](https://gtexportal.org/home/protectedDataAccess)
*   **Gene Expression:** The corresponding gene expression QTL data can be downloaded from: [https://www.gtexportal.org/home/downloads/adult-gtex/qtl](https://www.gtexportal.org/home/downloads/adult-gtex/qtl)

### Experiment Pipeline

1. After preprocessing the DNA sequences, generate zero-shot embeddings for each foundation model following gtex_[model_short_name].py in `job_scripts`.
2. Run `GTEX_cov_out.py` in `analysis` to regress out covariates from the gene expression data.
3. `GTEX_regression.py` to perform the final regression task, predicting gene expression from the processed DNA embeddings.
4. `GTEX_summary_results.py` to summarize and compare the final performance metrics across different foundation models.

## 4. Variant Effect Quantification

This section details the procedure for replicating our variant effect quantification results, including the pathogenic vs common variant analysis and the causal QTL analysis.

### Datasets

The primary datasets for this task are located in `data_processed/pathogenic` and `data_processed/causal`. This experiment also requires the hg38 reference genome fasta file, which we provide in `data_processed/TAD/hg38.ml.fa`.

### Experimental Pipeline

1.  First, navigate to the `analysis` directory, generate the required DNA sequences (i.e. with reference vs. alternate alleles) for the foundation models by running the `pathogenic_generate.py` script or `causal_generate.py`.

2.  Navigate to the `job_scripts` directory. For each foundation model, run the corresponding script to generate embeddings and calculate the distance metric between reference vs. alternate alleles. For example,
    ```bash
    python patho_[model_short_name].py
    ```
    or
    ```bash
    python causal_[model_short_name].py
    ```
    The outputs for each model will be saved to the `data_processed/pathogenic` or `data_processed/causal` with proper naming for later use.

3.  Navigate back to the `analysis` directory and run classification scripts to calculate the AUC and Cohen's d, measuring how good foundation models can generate embeddings to represent the distance between reference vs alternate alleles.
    ```bash
    python patho_classification.py
    ```
    or
    ```bash
    python causal_classification.py
    ```

## 5. TAD Region Recognition

This section describes the workflow for identifying Topologically Associating Domain (TAD) boundaries.

### Datasets

The data for this task is located in `data_processed/TAD/`. This includes data downloaded from (https://console.cloud.google.com/storage/browser/basenji_hic/insulation) and the reference genome files required for sequence generation.

### Experimental Pipeline

1.  **Preprocessing:** First, run `select_tads.py` to choose the TAD regions for analysis. Then, run `generate_seqs_from_tads.py` and `generate_seqs_random.py` to generate the positive (TAD boundary) and negative (random non-boundary) DNA sequences.

2.  **Embedding Generation:** In `job_scripts` directory, run `tad_ntv2.py` to generate the embeddings for the sequences.

3.  **Analysis and Plotting:** In `analysis` directory, and run `tad_interpret.py` to replicate the heatmap plot from our results.


## 6. Runtime Analysis

This section explains how to replicate the model runtime and throughput analysis.

1.  **Measure Runtimes:** In `job_scripts` directory, for each foundation model, run its corresponding runtime script.
    ```bash
    python runtime_[model_short_name].py
    ```
    You are encouraged to modify parameters within the script, such as `batch_size` and `sequence_length`, or substitute different model checkpoints to test various configurations. The results will be saved to the `runtimes/` folder.

2.  **Plotting:** To visualize the results and replicate our comparison plot, navigate to the `analysis` directory and run the plotting script `plot_runtime.py`.

## References
1. Zhou, Z. et al. DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome. Preprint at https://doi.org/10.48550/arXiv.2306.15006 (2024).
2. Dalla-Torre, H. et al. Nucleotide Transformer: building and evaluating robust foundation models for human genomics. Nat Methods 22, 287–297 (2025).
3. Nguyen, E. et al. HyenaDNA: long-range genomic sequence modeling at single nucleotide resolution. In Proceedings of the 37th International Conference on Neural Information Processing Systems (2023).
4. Schiff, Y. et al. Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling. Preprint at https://doi.org/10.48550/arXiv.2403.03234 (2024).
5. Sanabria, M., Hirsch, J., Joubert, P. M. & Poetsch, A. R. DNA language model GROVER learns sequence context in the human genome. Nat Mach Intell 6, 911–923 (2024).
6. Avsec, Ž. et al. Effective gene expression prediction from sequence by integrating long-range interactions. Nat Methods 18, 1196–1203 (2021).
