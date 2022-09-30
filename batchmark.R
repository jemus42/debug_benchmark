# Ensure current GitHub dependencies
# run renv::restore() to install dependencies as stored in renv.lock

library(batchtools)
library(mlr3)
library(mlr3batchmark) # renv::install("mlr-org/mlr3batchmark")
library(mlr3tuning)
library(mlr3learners)
library(mlr3pipelines)

# Have renv detect learner dependencies
if (FALSE) {
  library(ranger)
  library(xgboost)
}

# Settings
resample_outer <- rsmp("cv", folds = 10)
resample_inner <- rsmp("cv", folds = 5)
mymsr <- msr("classif.auc")
# mytrm <- trm("evals", n_evals = 50) # Trial mode
mytrm <- trm("evals", n_evals = 200)  # Serious mode
mytnr <- tnr("random_search")


# tasks
tasks <- list(
  #tsk("pima"),
  tsk("german_credit"),
  #tsk("penguins"),
  tsk("spam")
)

# Wrapping learners into auto_tuners with optional factor encoding for xgb
auto_tune <- function(learner, .encode = FALSE, ...) {
  search_space <- ps(...)

  if (.encode) {
    learner_graph <- po("encode", method = "treatment") %>>%
      po("learner", learner)
    learner <- as_learner(learner_graph)
  }

  AutoTuner$new(
    learner = learner,
    resampling = resample_inner,
    measure = mymsr,
    search_space = search_space,
    terminator = mytrm,
    tuner = mytnr
  )
}

# ranger
tuned_ranger <- auto_tune(
  learner = lrn("classif.ranger", predict_type = "prob", num.trees = 50, id = "ranger"),
  mtry.ratio = p_dbl(0.1, 1),
  min.node.size = p_int(1, 50),
  replace = p_lgl(),
  sample.fraction = p_dbl(0.1, 1)
)

# xgboost
tuned_xgboost <- auto_tune(
  learner = lrn("classif.xgboost", predict_type = "prob",
                nthread = 1, # Just to be safe
                id = "xgboost"
                ),
  .encode = TRUE,
  # Need to prefix params with learner id bc of pipeline
  xgboost.max_depth = p_int(1, 20),
  xgboost.subsample = p_dbl(0.1, 1),
  xgboost.colsample_bytree = p_dbl(0.1, 1),
  xgboost.nrounds = p_int(10, 5000),
  xgboost.eta = p_dbl(0, 1)
)

# xgboost: fixed depth
tuned_xgboost_fixdepth <- auto_tune(
  learner = lrn("classif.xgboost", predict_type = "prob",
                nthread = 1, # Just to be safe
                max_depth = 2, id = "xgboost_fixdepth"),
  .encode = TRUE,
  # Need to prefix params with learner id bc of pipeline
  xgboost_fixdepth.subsample = p_dbl(0.1, 1),
  xgboost_fixdepth.colsample_bytree = p_dbl(0.1, 1),
  xgboost_fixdepth.nrounds = p_int(10, 5000),
  xgboost_fixdepth.eta = p_dbl(0, 1)
)

# Benchmark design
learners <- list(
  tuned_ranger,
  tuned_xgboost,
  tuned_xgboost_fixdepth
)

design <- benchmark_grid(
  tasks = tasks,
  learners = learners,
  resamplings = list(resample_outer)
)

# Run with batchtools
reg_dir <- here::here("registry")
if (!dir.exists("registry")) dir.create("registry")

if (dir.exists(reg_dir)) {
  # Comment this line to prevent stored registry deletion on accident
  unlink(reg_dir, recursive = TRUE)
}

reg <- makeExperimentRegistry(reg_dir, seed = 230749)
batchmark(design, reg = reg)

# Overview of learner IDs
table(unwrap(getJobPars())[["learner_id"]])

# Submit
ids_xgb <- findExperiments(algo.pars = learner_id == "encodexgboost.tuned")
ids_xgb_fixdepth <- findExperiments(algo.pars = learner_id == "encode.xgboost_fixdepth.tuned")
ids_ranger <- findExperiments(algo.pars = learner_id == "ranger.tuned")

submitJobs(ids_xgb_fixdepth)

waitForJobs()
getStatus()
