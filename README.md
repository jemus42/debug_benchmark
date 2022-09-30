# Debug Benchmark

Benchmark including `ranger` and `xgboost`.

## Setup

- `batchtools.conf.R` batchtools config for running jobs locally (`Multicore` or `SSH`).
- `batchmark.R` contains learner definitions with parameter spaces and `batchtools` setup via `mlr3batchmark`.

## Context

The `xgboost` learner seemed to consume more CPU resources than it should have been allowed to.  
E.g. on a workstation with 48 threads available and and allocating 10 threads with batchtools, the load average would rise to around 48 after a few seconds.

It turns out that adding the following to `.Rprofile` helped:

```r
Sys.setenv(OMP_NUM_THREADS="1")
Sys.setenv(OPENBLAS_NUM_THREADS="1")
Sys.setenv(MKL_NUM_THREADS="1")
```
