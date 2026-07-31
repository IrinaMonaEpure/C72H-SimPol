#### Functions for binomial regressions ----
# Function to format output
formatModel.glm <- function(x) {
  model <- broom::tidy(x) %>%
    select(-statistic) %>%
    add_row(term = "N", estimate = length(x$residuals)) %>%
    add_row(term = "Deviance", estimate = x$deviance)
  attr(model, "meta_rows") <- c((nrow(model)-1):nrow(model))
  colnames(model) <- c("var", "b", "se", "p")
  return(model)
}

# Function to combine output
combineModels.glm <- function(models) {
  # Join together
  for (i in seq_along(models)) 
    if (i == 1) mat = models[[i]] else
      mat <- full_join(mat, models[[i]], by = c("term"), copy = T)
    
    # Relabel columns and write to file
    rowMeta1 <- nrow(models[[1]])-1
    rowMeta2 <- nrow(models[[1]])
    mat <- rbind(mat[-c(rowMeta1:rowMeta2), ], mat[c(rowMeta1:rowMeta2), ])
    rownames(mat) <- NULL
    colnames(mat) <- c("var", rep(c("b", "se", "p"), times = length(models)))
    
    return(mat)
}