# Bootstrap a single country ----
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
sd(results[,2], na.rm=T)
range(results[,2], na.rm=T)

# Write results to file
write_xlsx(
  as.data.frame(results), 
  path = "tables/robustness_sampling.xlsx")



#### Check how missings there were during calculation of CCA
# How many 0, 1 or 2 missing
df_subset <- df %>%
  select(lrscale:anti_libertarianism)
temp <- apply(df_subset, 1, function (x) sum(is.na(x)))
table(temp)
table(temp)/sum(table(temp))

# How many missings per calculated correlation
temp1 <- map_dfr(countries_cca, function (x) {
  temp <- x[["nvars"]]
  temp[lower.tri(temp)] <- NA
  table(temp)
})
colnames(temp1) <- c("m16", "m17", "m18", "m19", "m20")
temp2 <- as_tibble(temp1 / rowSums(temp1)) %>%
  select(ends_with("Freq"))
temp2
colMeans(temp2)*100



#### Other clustering algorithms
cbind(class_names, 
      leading.eigenvector.community(grps$network)$membership,
      fastgreedy.community(grps$network)$membership,
      edge.betweenness.community(grps$network, directed = F)$membership,
      walktrap.community(grps$network)$membership)

border_cases <- cbind(leading.eigenvector.community(grps$network)$membership,
                        walktrap.community(grps$network)$membership)
rownames(border_cases) <- class_names

border_cases <- rownames(border_cases[which(border_cases[,1] != border_cases[,2]),])
border_cases

grps$group_membership

altgr_membership <- grps$group_membership
altgr_membership[border_cases] <- 3
altgr_membership

altgr_group1 <- names(altgr_membership[which(altgr_membership == 1)])
altgr_group2 <- names(altgr_membership[which(altgr_membership == 2)])
altgr_group3 <- names(altgr_membership[which(altgr_membership == 3)])


### Group descriptives (table 2)
altgr_descriptives <- matrix(data = NA, nrow = 4, ncol = 3, 
                             dimnames = list(
                               c("Number of belief systems", 
                                 "Between-belief system density", 
                                 "Similarities", 
                                 "Mean of belief system densities"),
                               c("Group 1", "Group 2", "Group 3")))

# Number of classes per group
altgr_descriptives["Number of belief systems", 1] <- length(classes[altgr_group1])
altgr_descriptives["Number of belief systems", 2] <- length(classes[altgr_group2])
altgr_descriptives["Number of belief systems", 3] <- length(classes[altgr_group3])

# Descriptives about similarities between classes within and between the groups
altgr_descriptives["Similarities",1] <- mean(cca_similarities[altgr_group1,altgr_group1], na.rm = T)
altgr_descriptives["Similarities",2] <- mean(cca_similarities[altgr_group2,altgr_group2], na.rm = T)
altgr_descriptives["Similarities",3] <- mean(cca_similarities[altgr_group3,altgr_group3], na.rm = T)

altgr_descriptives["Between-belief system density",1] <- mean(classes_pearson[altgr_group1,altgr_group1], na.rm = T)
altgr_descriptives["Between-belief system density",2] <- mean(classes_pearson[altgr_group2,altgr_group2], na.rm = T)
altgr_descriptives["Between-belief system density",3] <- mean(classes_pearson[altgr_group3,altgr_group3], na.rm = T)

# Mean density of the classes
altgr_descriptives["Mean of belief system densities", 1] <- mean(class_descriptives[altgr_group1, "mean"])
altgr_descriptives["Mean of belief system densities", 2] <- mean(class_descriptives[altgr_group2, "mean"])
altgr_descriptives["Mean of belief system densities", 3] <- mean(class_descriptives[altgr_group3, "mean"])

altgr_descriptives

# Cultural-economic correlations
print(psych::describe(cultecon_corr[altgr_group1, "value"]), digits = 3)
print(psych::describe(cultecon_corr[altgr_group2, "value"]), digits = 3)
print(psych::describe(cultecon_corr[altgr_group3, "value"]), digits = 3)


# Cluster analysis without non-EU countries ----
countries_nonEU <- c("CH", "IL", "IS", "NO", "RU")
countries_EU <- countries[!countries %in% countries_nonEU]

class_names_EU <- vector()
class_names_nonEU <- vector()
for (x in class_names) {
  if (str_split(x, "_", simplify = T)[1] %in% countries_EU) {
    class_names_EU <- append(class_names_EU, x)
  } else{
    class_names_nonEU <- append(class_names_nonEU, x)
  }
}
class_names_EU
class_names_nonEU

## Convert to network
classes_pearson_EU <- classes_pearson[class_names_EU, class_names_EU]

diag(classes_pearson_EU) <- 1

network_EU <- graph.adjacency(
  abs(classes_pearson_EU),
  mode = "undirected",
  weighted = TRUE,
  diag = FALSE
)

## Define group membership
groups_EU <- leading.eigenvector.community(network_EU)$membership
names(groups_EU) <- class_names_EU
groups_EU
unique(groups_EU)

## Compare Euroscepticism correlation between EU and non-EU
temp <- map(classes, \(x) data.frame(rn = x["euroscepticism",]))
temp1 <- rowMeans(list_cbind(temp[class_names_EU]))
temp2 <- rowMeans(list_cbind(temp[class_names_nonEU]))
cbind(temp1, temp2)


# CCA analyses without lrscale ----
# List of results
nolrscale_cca <- vector("list", length(countries))

# Run CCA
cl <- makeCluster(12) # better: detectCores() - 1
system.time({
  registerDoParallel(cl)
  nolrscale_cca <- foreach(
    i = seq_along(countries), 
    .packages=c("tidyverse")
  ) %dopar% {
    df_subset <- df %>% 
      filter(country == countries[[i]]) %>% # Filter by country
      select(gender_inequality:anti_libertarianism) # Select only beliefs, without lrscale
    nolrscale_cca[i] <- bscca(df_subset, zero.action = "ownclass") # Perform modified CCA
  }})
stopCluster(cl)

# Save to file
save(nolrscale_cca, file = "data/nolrscale_cca.RData")

# Remove degenerate classes and reorder membership data
# cca will probably still say that there is one more class, but it will only contain the non-degenerate classes
for (i in c(1:length(countries))) {
  degenerate <- 0
  for (ii in 1:length(nolrscale_cca[[i]]$modules)) {
    k <- ii - degenerate
    if (nolrscale_cca[[i]]$modules[[k]]$degenerate == T) { 
      # Class is degenerate so shift membership data one place
      nolrscale_cca[[i]]$membership <- if_else(nolrscale_cca[[i]]$membership == k, 
                                               NA_real_, nolrscale_cca[[i]]$membership)
      nolrscale_cca[[i]]$membership <- if_else(nolrscale_cca[[i]]$membership > k, 
                                               nolrscale_cca[[i]]$membership - 1, nolrscale_cca[[i]]$membership)
      nolrscale_cca[[i]]$modules[[k]] <- NULL
      degenerate <- degenerate + 1
    }
  }
}
rm(degenerate)

## Inspecting the results ----
# How many classes per country
map_vec(
  1:23,
  \(x) length(nolrscale_cca[[x]]$modules) - length(countries_cca[[x]]$modules)
)

# How many mismatches per country
map_vec(
  1:23,
  \(x) sum(nolrscale_cca[[x]]$membership != countries_cca[[x]]$membership, na.rm = T)
)


## Calculate between-belief system correlations ----
### Create list of belief system correlation matrices (like -classes-)
nolr_classes <- list()
nolr_class_names <- vector()
k <- 1
for (i in c(1:length(countries))) {
  for (ii in 1:length(nolrscale_cca[[i]]$modules)) {
    nolr_classes[[k]] <- nolrscale_cca[[i]]$modules[[ii]]$cormat # save class to list
    nolr_class_names <- c(nolr_class_names, paste0(countries[[i]], "_", ii)) # save class names: country_class#

    k <- k + 1
  }
}
nolr_classes <- map(nolr_classes, \(x) { diag(x) <- NA; x })

### Calculate correlation between original and nolr belief system cormats
temp <- map_vec(
  seq(length(nolr_classes)),
  \(x) cor(
    classes[[x]][-1,-1][lower.tri(matrix(nrow = 19, ncol = 19))], 
    nolr_classes[[x]][lower.tri(matrix(nrow = 19, ncol = 19))]
  )
)
