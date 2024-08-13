# DNA Foundation Models Benchmarking
## Introduction
This repo is for generating the results of DNA foundation models benchmarking.

Please cite the following manuscript for using DNAm models built and association results by our work:  

> Feng, H., Wei, P., Wu, C., 2024. Benchmarking DNA Foundation Models for Genomic Sequence Classification. Under Review.

## Workflow
<div style="text-align: center;">
  <img src="https://github.com/ChongWuLab/dna_foundation_benchmark/blob/main/Fig1.png" width=50% height=50%>
</div>

## Genetic data processing

## References

After running scripts in `./job_scripts`, there should be folders in `/results_final` named dnabert2, dnabert2_meanpool, ntv2, ntv2_meanpool, hyena, hyena_meanpool, each storing the performance of all 56 datasets.

Then by running `./results_final/combine.py`, organized results named like "final_dnabert2.csv", "final_dnabert2_meanpool.csv", etc. will be generated.

The plots and tables in the manuscript can then be generated by `./results_final/process.py`, `./results_final/plot_radar.py`, `./results_final/plot_box.py` and `./results_final/plot_runtime.py`.
