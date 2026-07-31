### Just run the whole file - it'll create new functions that can be called by the main function -bscca- 
# Not pretty, but it works

#### Own function
# function to create matrix
bscor_matrix <- function(x, iter = 1000) {
  n <- ncol(x)
  results <- matrix(0, nrow = nrow(x), ncol = nrow(x))
  x <- t(x)
  for (i in c(1:iter)) {
    recode_vars <- sample(c(0, 1), n, replace = T)
    x <- abs(recode_vars - x)
    results <- results + cor(x, use = "pairwise.complete.obs")
  }
  results <- results / iter
  return(results)
}

#### Rebuilding CCA functions
# Main cca function
bscca <- function (dtf, filter.significance = TRUE, filter.value = 0.01, 
                   zero.action = c("drop", "ownclass"), verbose = TRUE) 
{
  if (verbose) 
    echo <- cat
  else echo <- c
  if (packageVersion("igraph") < 0.7) {
    warning(paste("Your igraph version is ", as.character(packageVersion("igraph")), 
                  ". CCA produces more accurate results with igraph >= 0.7.\n", 
                  sep = ""))
  }
  cormat <- bs.make.cormat(dtf, zero.action) # change .make.cormat -> bs.make.cormat
  if (filter.significance == TRUE) {
    echo("Filtering out correlations for which Pr(|r| != 0) > ", 
         filter.value, "\n")
    # Calculate n_variables per correlation
    r <- !is.na(t(dtf))
    N.vars <- t(r) %*% r
    cormat <- bs.filter.insignif(cormat, N.vars, pcutoff = filter.value) # change .filter.insignif -> bs.filter.insignif & ncol(dtf) -> N.vars
  }
  else {
    echo("Not filtering significances.\n")
  }
  graph <- bs.cormat.to.igraph(cormat, absolute.value = TRUE) # change .cormat.to.igraph -> bs.cormat.to.igraph
  comm <- igraph::leading.eigenvector.community(graph)
  modules <- bs.separate(attr(cormat, "dtf"), comm$membership) # change .separate -> bs.separate
  val <- list(membership = comm$membership, modules = modules, 
              cormat = cormat)
  class(val) <- "cca"
  #echo(paste(capture.output(print(val)), collapse = "\n"), # Put in comment cause it gives an error? work fine like this
  #     "\n")
  return(invisible(val))
}

# Make cormat function
bs.make.cormat <- function (dtf, zero.action) 
{
  if (!all(sapply(dtf, is.numeric))) 
    dtf2 <- data.frame(sapply(dtf, as.numeric))
  else dtf2 <- dtf
  zeros <- which(apply(dtf2, 1, var) <= 1e-09)
  if (zero.action[1] == "drop" & (length(zeros) > 0)) {
    dtf2 <- dtf2[-zeros, ]
  }
  rv <- abs(bscor_matrix(dtf2)) # change cor(t()) -> bscor_matrix()
  attributes(rv)$zeros <- zeros
  attributes(rv)$zero.action <- zero.action[1]
  attributes(rv)$dtf <- dtf2
  if ((zero.action[1] == "ownclass") & length(zeros) > 0) {
    rv[zeros, ] <- 0
    rv[, zeros] <- 0
    rv[zeros, zeros] <- 1
  }
  diag(rv) <- 0
  return(rv)
}

# Filter insignificance function
bs.filter.insignif <- function (corr, N.vars, pcutoff = 0.05) 
{
  corr <- abs(corr)
  if (any(diag(corr) != 0)) 
    stop("Non-zero elements on the diagonal. diag(corr) <- 0 before running this function.")
  suppressWarnings(tvalues <- corr * sqrt((N.vars - 2)/(1 - 
                                                          corr^2)))
  if (any(is.infinite(tvalues))) {
    tvalues[is.infinite(tvalues)] <- 9999
  }
  cutoff <- abs(qt(pcutoff/2, N.vars))
  isolates.pre <- sum(apply(corr, 1, sum) == 0)
  corr[tvalues < cutoff] <- 0
  isolates.post <- sum(apply(corr, 1, sum) == 0)
  if (isolates.post > isolates.pre) {
    warn1 <- paste("Significance filtering left", isolates.post - 
                     isolates.pre, "rows with no non-zero ties. The CCA result will contain at least one small degenerate class.")
    warning(warn1)
  }
  return(corr)
}


# Cormat to igraph function
bs.cormat.to.igraph <- function (corr, absolute.value = TRUE) 
{
  if (absolute.value) 
    corr <- abs(corr)
  diag(corr) <- 0
  graph <- igraph::graph.adjacency(corr, mode = "undirected", 
                                   weighted = TRUE, diag = FALSE)
  return(graph)
}


# Separate modules function
bs.separate <- function (dtf, membership) 
{
  ids <- sort(unique(membership))
  modules <- list()
  if (is(dtf, "matrix")) {
    rownames(dtf) <- NULL
    dtf <- data.frame(dtf)
  }
  for (i in 1:length(ids)) {
    curmod <- list()
    class(curmod) <- "cca.module"
    curmod$dtf <- dtf[membership == ids[i], ]
    curmod$cormat <- cor(curmod$dtf, use = "pairwise.complete.obs")
    curmod$degenerate <- any(is.na(curmod$cormat))
    modules[[i]] <- curmod
  }
  return(modules)
}
                       
                                 
  
