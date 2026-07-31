#### Analyses ----
# Calculate descriptives for each class
class_descriptives <- data.frame(matrix(ncol = 9, nrow = total_classes))
colnames(class_descriptives) <- c("country", "class", "n", "n_share", "mean", "sd", "median", "min", "max")
for (i in c(1:length(classes))) {
  class_descriptives$country[i] <- str_split(names(classes[i]), "_", simplify = T)[1]
  class_descriptives$class[i] <- as.numeric(str_split(names(classes[i]), "_", simplify = T)[2])
  class_descriptives$n[i] <- length(df$id[which(df$class_id == i)])
  class_descriptives$n_share[i] <- class_descriptives$n[i] / length(df$id[which(df$country == class_descriptives$country[i])])
  class_descriptives$mean[i] <- mean(abs(classes[[i]]), na.rm = T)
  class_descriptives$sd[i] <- sd(abs(classes[[i]]), na.rm = T)
  class_descriptives$median[i] <- median(abs(classes[[i]]), na.rm = T)
  class_descriptives$min[i] <- min(classes[[i]], na.rm = T)
  class_descriptives$max[i] <- max(classes[[i]], na.rm = T)
}
rownames(class_descriptives) <- class_names
class_descriptives

# Classify cca correlations by direction (with cutoff of 0.05)
classes_classified <- lapply(classes, function(y) {
  y[] <- sapply(y, function(x) { 
    if (is.na(x)) NA_real_ else if (abs(x) > 0.05) sign(x) else 0
  })
  return(y)
})
classes_classified[[1]]


# Check amount of correlational similarities between classes both over entire matrix and over each row
x <- y <- z <- 1
cca_similarities = matrix(data=0, nrow=total_classes, ncol=total_classes)
cca_rowsimilarities = as.data.frame(matrix(nrow=total_classes*total_classes-total_classes, ncol=total_vars))
for (i in seq_along(classes)) { 
  for (ii in seq_along(classes)) { 
    if (ii > i) {
      cca_rowsimilarities[z,1:2] <- c(class_names[i], class_names[ii])
      cca_rowsimilarities[z+1,1:2] <- c(class_names[ii], class_names[i])
      for (k in c(1:nrow(classes[[i]]))) {
        row_similarities <- 0
        for (l in c(1:ncol(classes[[i]]))) { 
          if (!is.na(classes_classified[[i]][k,l]) & 
              classes_classified[[i]][k,l] == classes_classified[[ii]][k,l]) { 
            # Direction of significant coefficients are identical and not NA
            cca_similarities[x,y] <- cca_similarities[x,y] + 1
            cca_similarities[y,x] <- cca_similarities[y,x] + 1
            row_similarities <- row_similarities + 1
          } 
        }
        cca_rowsimilarities[z,k+2] <- row_similarities / (total_vars-1)
        cca_rowsimilarities[z+1,k+2] <- row_similarities / (total_vars-1)
      }
      z <- z + 2
    }
    x <- x+1
  }
  x <- 1
  y <- y+1
}
diag(cca_similarities) <- NA
colnames(cca_rowsimilarities) <- c("country1", "country2", var_names)
colnames(cca_similarities) <- rownames(cca_similarities) <- class_names
cca_similarities <- cca_similarities / sum(!is.na(classes[[1]]))
cca_similarities
head(cca_rowsimilarities)


##### Correlations, convert to network and find groups -----
# Pearson correlations between classes (with Mantel test function)
x <- y <- 1
classes_pearson <- matrix(nrow=total_classes, ncol=total_classes, dimnames=list(class_names,class_names))
temp <- lapply(classes, function(x) { diag(x) <- 1; return(x)})
for (i in c(1:length(classes))) { 
  for (ii in c(1:length(classes))) { 
    classes_pearson[x,y] <- evolqg::MantelCor(temp[[i]], temp[[ii]], permutations = 1)[1]
    x <- x+1
  }
  x <- 1
  y <- y+1
}
rm(temp)

# Intraclass correlation of countries over class correlations
temp <- lapply(rep(class_names, each = total_classes), 
               function (x) {
                 str_split(x, "_")[[1]][1]
               })
diag(classes_pearson) <- NA
temp <- data.frame("r" = unlist(as.list(classes_pearson)), 
                   "country" = factor(unlist(temp))) %>% 
  drop_na(r) 

icc <- performance::icc(lmer(r ~ 1 + (1 | country),
                             data = temp))
icc



