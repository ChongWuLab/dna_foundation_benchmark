# DNA Foundation Models Benchmarking
## Introduction
This repo is for generating the results of DNA foundation models benchmarking.

Please cite the following manuscript for using DNAm models built and association results by our work:  


> Feng, Haonan, Lang Wu, Bingxin Zhao, Chad Huff, Jianjun Zhang, Jia Wu, Lifeng Lin, Peng Wei, and Chong Wu. "[https://www.biorxiv.org/content/10.1101/2024.08.16.608288v1](Benchmarking DNA Foundation Models for Genomic Sequence Classification)." bioRxiv (2024): 2024-08.

Feng, Haonan, Lang Wu, Bingxin Zhao, Chad Huff, Jianjun Zhang, Jia Wu, Lifeng Lin, Peng Wei, and Chong Wu. [https://www.biorxiv.org/content/10.1101/2024.08.16.608288v1](Benchmarking DNA Foundation Models for Genomic Sequence Classification). bioRxiv (2024): 2024-08.


## Workflow
<p align="center">
  <img src="https://github.com/ChongWuLab/dna_foundation_benchmark/blob/main/Fig1.png" width=50% height=50%>
</p>

## Genomic data processing
We collected the preprocessed datasets of genomic tasks from published works with DNA sequences and corresponding labels. We further processed these datasets for train-test split, and the processed datasets can be downloaded **here**. We maintained the training and testing split of datasets from their original works if available; otherwise, we randomly split the samples into a ratio of 7:3 for training and testing. Detailed descriptions of each dataset can be found in our supplementary materials **Dataset naming** section.

## Zero-shot embedding generation
After running scripts `job_scripts/inference_dnabert2.py`, `job_scripts/inference_ntv2.py`, `job_scripts/inference_hyena.py`, with arguments specifying name of dataset folder, output pooling method and maximum sequence length, csv files will be generated containing the zero-shot embeddings of sequences for training and testing data.

## Classification performance report
The scripts `job_scripts/classify_dnabert2.py`, `job_scripts/classify_ntv2.py`, `job_scripts/classify_hyena.py` will take the zero-shot embedding csv files as input, and report the classification performances of each model. Then by running `results_final/combine.py`, organized results named like "final_dnabert2.csv", "final_dnabert2_meanpool.csv", etc. will be generated in `results_final` folder.

## Test of significance
`job_scripts/delong.py` is used to compare the AUC of each model on each dataset.

## Organized results and figures
`results_final/process.py`: Generate the tables to compare (1) model performances on 4 categories of tasks, (2) model performances on multiclass classification tasks, (3) model performances with summary token pooling and mean pooling.\
`results_final/plot_radar.py`: Generate the radar plot Supplementary figure 1.\
`results_final/plot_box.py`: Generate the boxplots of summary token pooling versus mean pooling.\
`results_final/plot_runtime.py`: Generate the runtime plot.

## References
1. Zhou, Z. et al. DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome. Preprint at https://doi.org/10.48550/arXiv.2306.15006 (2024).
2. Dalla-Torre, H. et al. The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics. Preprint at https://doi.org/10.1101/2023.01.11.523679 (2023).
3. Nguyen, E. et al. HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution. Preprint at https://doi.org/10.48550/arXiv.2306.15794 (2023).
