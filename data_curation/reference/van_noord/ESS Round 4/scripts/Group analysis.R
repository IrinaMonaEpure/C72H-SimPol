##### Convert between-belief system correlations to network and find groups ----
# Create environment for all vars/data/results of community comparison
grps <- new.env()
grps$title <- "groups"

# Convert to network
diag(classes_pearson) <- 1
grps$network <- graph.adjacency(abs(classes_pearson), mode = "undirected",
                                weighted = TRUE, diag = FALSE)

# Define group membership
grps$group_membership <- leading.eigenvector.community(grps$network)$membership
names(grps$group_membership) <- class_names
grps$group_membership
unique(grps$group_membership)

V(grps$network)$community <- grps$group_membership

# Define groups
grps$group1 <- names(grps$group_membership[grps$group_membership == 1])
grps$group2 <- names(grps$group_membership[grps$group_membership == 2])

grps$groups_names <- c("group1", "group2")
grps$group1
grps$group2

# Add class community membership to df
df <- df %>% 
  unite(class_name, c("country", "cca_membership"), remove = F) %>%
  mutate(class_community = case_when(
    class_name %in% grps$group1 ~ 1,
    class_name %in% grps$group2 ~ 2),
    dummy_community1 = 2 - class_community,
    dummy_community2 = class_community - 1)

### Group descriptives (table 2)
grps$descriptives <- matrix(data = NA, nrow = 4, ncol = 2, 
                            dimnames = list(
                              c("Number of belief systems", 
                                "Between-belief system density", 
                                "Similarities", 
                                "Mean of belief system densities"),
                              c("Group 1", "Group 2")))

# Number of classes per group
grps$descriptives["Number of belief systems", 1] <- length(classes[grps$group1])
grps$descriptives["Number of belief systems", 2] <- length(classes[grps$group2])

# Descriptives about similarities between classes within and between the groups
grps$descriptives["Similarities",1] <- mean(cca_similarities[grps$group1,grps$group1], na.rm = T)
grps$descriptives["Between-belief system density",1] <- mean(classes_pearson[grps$group1,grps$group1], na.rm = T)

grps$descriptives["Similarities",2] <- mean(cca_similarities[grps$group2,grps$group2], na.rm = T)
grps$descriptives["Between-belief system density",2] <- mean(classes_pearson[grps$group2,grps$group2], na.rm = T)

# Mean density of the classes
grps$descriptives["Mean of belief system densities", 1] <- mean(class_descriptives[grps$group1, "mean"])
grps$descriptives["Mean of belief system densities", 2] <- mean(class_descriptives[grps$group2, "mean"])

grps$descriptives


##### Find average belief system per group ----
# Get average correlation matrix
for (i in c(1:length(grps$group1))) {
  if (i == 1) temp <- classes[[grps$group1[[i]]]]
  else temp <- temp + classes[[grps$group1[[i]]]]
}
grps$average1 <- temp / length(grps$group1)

for (i in c(1:length(grps$group2))) {
  if (i == 1) temp <- classes[[grps$group2[[i]]]]
  else temp <- temp + classes[[grps$group2[[i]]]]
}
grps$average2 <- temp / length(grps$group2)

diag(grps$average1) <- NA
diag(grps$average2) <- NA


#### FA ----
# Factor analysis
temp1 <- grps$average1
temp2 <- grps$average2
diag(temp1) <- diag(temp2) <- 1
fa_group1 <- psych::fa(temp1, nfactors = 4)
fa_group1
fa_group2 <- psych::fa(temp2, nfactors = 2)
fa_group2

# Predict factor scores, including for those with missing values
df_subset <- df %>%
  dplyr::select(lrscale:anti_libertarianism)

temp1 <- predict(fa_group1, df_subset, missing = TRUE)
colnames(temp1) <- c("economic1", "drop1", "drop2", "cultural1")
temp1 <- temp1[, c("economic1", "cultural1")]

temp2 <- predict(fa_group2, df_subset, missing = TRUE)
colnames(temp2) <- c("economic2", "cultural2")

# Bind resulting latent variables to df
df <- cbind(df, temp1, temp2)

# Correlations for each belief system separately
temp1 <- temp2 <- vector()
for (i in c(1:total_classes)) {
  temp1[i] <- with(df[which(df$class_id == i), ], 
                   cor(cultural1, economic1))
  temp2[i] <- with(df[which(df$class_id == i), ], 
                   cor(cultural2, economic2))
}
names(temp1) <- names(temp2) <- class_names

# Put in a dataframe
cultecon_corr <- as.data.frame(rbind(cbind(value = temp1[grps$group1], group = 1),
                                     cbind(value = temp2[grps$group2], group = 2)))
cultecon_corr <- cultecon_corr[order(row.names(cultecon_corr)), ]
cultecon_corr$group <- factor(cultecon_corr$group, levels = c(1,2), labels = c("Group 1", "Group 2"))


# Describe correlations
psych::describe(cultecon_corr[grps$group1, "value"])
psych::describe(cultecon_corr[grps$group2, "value"])

# strongest correlations in each group 
cultecon_corr %>% filter(group == "Group 1") %>% arrange(desc(abs(value))) %>% slice(1:5)
cultecon_corr %>% filter(group == "Group 2") %>% arrange(desc(abs(value))) %>% slice(1:5)

# Correlation between correlation and density of rest of belief system
items <- c("lrscale", "anti_egalitarianism", "benefits_eco", "anti_interventionism", "government_spending", "gender_inequality", "anti_lgbt", "anti_immigration", "authoritarianism", "age_prejudice")
temp <- unlist(lapply(classes, function (x) { mean(abs(x[-which(var_names %in% items), 
                                                         -which(var_names %in% items)]), na.rm = T) }))
cor(temp[grps$group1], cultecon_corr[grps$group1, "value"])
cor(temp[grps$group2], cultecon_corr[grps$group2, "value"])

# most polarized countries
temp <- cultecon_corr %>% 
  mutate(class = rownames(cultecon_corr)) %>%
  mutate(country = str_split(class, "_", simplify = T)[,1],
         classn = str_split(class, "_", simplify = T)[,2]) %>%
  group_by(country) %>%
  summarize(min = min(value), max = max(value)) %>%
  mutate(diff = max-min) %>%
  arrange(desc(diff))

print(temp, n = total_countries)
mean(temp$diff)