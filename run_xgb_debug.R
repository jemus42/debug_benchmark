# Isolated training on xgb without batchtools
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(mlr3pipelines)

learner_in <- lrn("classif.xgboost", predict_type = "prob")

learner_graph <- po("encode", method = "treatment") %>>%
  po("learner", learner_in)

learner <- as_learner(learner_graph)

at <- AutoTuner$new(
  learner = learner,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.auc"),
  search_space = ps(
    classif.xgboost.max_depth = p_int(1, 20),
    classif.xgboost.subsample = p_dbl(0.1, 1),
    classif.xgboost.colsample_bytree = p_dbl(0.1, 1),
    classif.xgboost.nrounds = p_int(10, 5000),
    classif.xgboost.eta = p_dbl(0, 1)
  ),
  terminator = trm("evals", n_evals = 100),
  tuner = tnr("random_search")
)

at$train(tsk("german_credit"))

