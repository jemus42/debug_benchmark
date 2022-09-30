cluster.functions <- makeClusterFunctionsSSH(list(Worker$new("localhost", ncpus = 10)))

# cluster.functions <- makeClusterFunctionsMulticore(ncpus = 10)
