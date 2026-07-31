### Bootstrap a single country
bootstrap_country <- function(country_id, iterations = 100, sample_size = 0.5) {
  # Take original cormat
  cormat_old <- countries_cca[[country_id]]$cormat
  
  # Set options
  nobs <- length(country_membership[[country_id]]) # Number of observation in the class
  sample_size <- nobs * sample_size # Sample size to take (default = 0.5 = half of the original N)
  
  # Create dataframe for results
  results <- data.frame(matrix(nrow = iterations, 
                               ncol = 3, 
                               dimnames = list(NULL, 
                                               c("N", "CramersV", "n_classes"))))
  
  # Take old membership, replace NA with 99
  old_membership <- country_membership[[country_id]]
  old_membership[which(is.na(old_membership))] <- 99
  
  # Bootstraps
  for (i in 1:iterations) {
    sample <- sample(c(1:nobs), replace = F, size = sample_size)
    cormat_new <- cormat_old[sample, sample]
    
    zeros <- which(apply(cormat_new, 1, var) <= 1e-09)
    if (length(zeros) > 0) sample <- sample[-zeros] # remove zero variance
    cormat_new <- cormat_old[sample, sample]
    
    network <- graph.adjacency(cormat_new, mode="undirected",
                               weighted = TRUE, diag = FALSE)
    
    skip_to_next <- FALSE # handling errors with trycatch
    tryCatch({
      membership <- leading.eigenvector.community(network)$membership
      
      results[i,"N"] <- length(sample)
      results[i,"CramersV"] <- cramerV(table(membership, old_membership[sample]))
      results[i,"n_classes"] <- length(table(membership))
    }, error = function(e) { print(paste0(i, ": ", e)); skip_to_next <<- TRUE})
    if(skip_to_next) { next }  
  }
  return(results)
}

### Loop through all countries and re-analyze class membership with smaller sample using function above
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
results <- foreach(i = seq_along(countries), .combine = rbind, .packages = c("igraph", "rcompanion")) %dopar% {
  temp <- bootstrap_country(i, iterations = 100, sample_size = 0.5)
  cbind(i, 
        mean(temp[which(temp[,2] != Inf), 2], na.rm=T), 
        length(which(temp[,2] == Inf)),
        t(quantile(temp[,2], c(.05, .5, .95), na.rm = T)))
}
stopCluster(cl) 

# Show results
results
apply(results, 2, mean, na.rm=T)

# Write results to file
write_xlsx(
  as.data.frame(results), 
  path = "tables/robustness_sampling.xlsx")
