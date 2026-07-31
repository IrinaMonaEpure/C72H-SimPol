# Plots ----
## corrplots  ----
for (i in class_names) {
  pdf(
    file = paste0("corrplots/", i, ".pdf"),
    width = 8.39, height = 8.39
  )
  corrplot.mixed(
    classes[[i]], 
    tl.pos = "lt", tl.cex = 0.8, number.cex = 0.3, lower.col = "black"
  )
  dev.off()
}

## qgraphs ----
for (i in class_names) {
  pdf(
    file = paste0("qgraphs/", i, ".pdf"),
    width = 8.39, height = 8.39
  )
  qgraph::qgraph(
    classes[[i]],
    layout = "spring",
    title = paste0("Class: ", i),
    maximum = 1
  )
  dev.off()
}


# Analyses ----
## Calculate descriptives for each class ----
get_descriptives <- function (class_id) {
  class            <- classes[[class_id]]
  abs_class        <- abs(class)
  class_name_split <- str_split(class_id, "_", simplify = T)
  country_code     <- class_name_split[1]
  n                <- class_membership[[class_id]]
  
  data.frame(
    country   = country_code,
    class     = as.numeric(class_name_split[2]),
    n,
    n_share   = n / length(df$id[which(df$country == country_code)]),
    mean      = mean(abs_class, na.rm = T),
    sd        = sd(abs_class, na.rm = T),
    median    = median(abs_class, na.rm = T),
    min       = min(class, na.rm = T),
    max       = max(class, na.rm = T),
    row.names = class_id
  )
}
class_descriptives <- map_dfr(class_names, get_descriptives)
head(class_descriptives)

## Classify cca correlations by direction (with cutoff of 0.05) ----
classify_correlation <- function (correlation) { 
  if (is.na(correlation)) NA_real_ else if (abs(correlation) > 0.05) sign(correlation) else 0
}
classes_classified <- map(
  classes, 
  \(class) {
    class[] <- map_dbl(class, classify_correlation)
    return(class)
  }
)
classes_classified[[1]]


## Create dataframe with all combinations of belief system names to loop over ----
combos <- expand.grid(class_names, class_names)
colnames(combos) <- c("class1", "class2")

## Check amount of correlational similarities between classes over entire matrix ----
total <- sum(!is.na(classes_classified[[1]]))
cca_similarities <- pmap_dbl(
  combos, 
  \(class1, class2) {
    sum(classes_classified[[class1]] == classes_classified[[class2]], na.rm = T) / total
  }
)
cca_similarities <- matrix(cca_similarities, 
                           nrow = total_classes, 
                           dimnames = list(class_names, class_names))

## Check amount of correlational similarities between classes over each row / variable ----
calc_rowsimilarities <- function(class1, class2) {
  rowsimilarities <- map_dbl(
    var_names,
    \(row) {
      sum(classes_classified[[class1]][row, ] == classes_classified[[class2]][row, ], na.rm = T)
    })
  names(rowsimilarities) <- var_names
  means = rowsimilarities / (total_vars-1)
  return(means)
}

cca_rowsimilarities <- pmap_dfr(combos, calc_rowsimilarities) %>%
  cbind(combos, .) |> 
  filter(class1 != class2)
head(cca_rowsimilarities)

## Pearson correlations between classes ----
classes_lower_triangle <- map(classes, \(class) class[lower.tri(class)])
classes_pearson <- pmap_dbl(
  combos, 
  \(class1, class2) {
    cor(classes_lower_triangle[[class1]], classes_lower_triangle[[class2]])
  }
)
classes_pearson <- matrix(
  classes_pearson, 
  nrow = total_classes, 
  dimnames=list(class_names,class_names)
)
rm(classes_lower_triangle)

## Intraclass correlation of countries over class correlations ----
temp <- rep(
  str_split(class_names, "_", simplify = T)[, 1], 
  each = total_classes
)
diag(classes_pearson) <- NA
temp <- data.frame(
  "correlation" = as.vector(classes_pearson), 
  "country" = temp
) |>
  drop_na(correlation) 

icc <- performance::icc(
  lmer(correlation ~ 1 + (1 | country),
       data = temp)
)
icc 


