# Benchmarking DNA Foundation Models for Genomic Sequence Classification

This repo is for generating the results of DNA foundation models benchmarking.

After running scripts in `./job_scripts`, there should be folders in `/results_final` named dnabert2, dnabert2_meanpool, ntv2, ntv2_meanpool, hyena, hyena_meanpool, each storing the performance of all 56 datasets.

Then by running `./results_final/combine.py`, organized results named like "final_dnabert2.csv", "final_dnabert2_meanpool.csv", etc. will be generated.

The plots and tables in the manuscript can then be generated by `./results_final/process.py`, `./results_final/plot_radar.py`, `./results_final/plot_box.py` and `./results_final/plot_runtime.py`.
