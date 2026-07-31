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
grps$descriptives <- matrix(
  data = NA, 
  nrow = 4, 
  ncol = 2, 
  dimnames = list(
    c("Number of belief systems", 
      "Between-belief system density", 
      "Similarities", 
      "Mean of belief system densities"),
    c("Group 1", "Group 2"))
)

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
total <- matrix(0, nrow = 20, ncol = 20)
for (class in classes[grps$group1]) total <- total + class
grps$average1 <- total / length(grps$group1)

total <- matrix(0, nrow = 20, ncol = 20)
for (class in classes[grps$group2]) total <- total + class
grps$average2 <- total / length(grps$group1)

diag(grps$average1) <- NA
diag(grps$average2) <- NA

# Calculate standard deviation
total <- matrix(0, nrow = 20, ncol = 20)
for (class in classes[grps$group1]) total <- total + (class - grps$average1)^2
grps$sd_average1 <- sqrt(total / length(grps$group1))

total <- matrix(0, nrow = 20, ncol = 20)
for (class in classes[grps$group2]) total <- total + (class - grps$average2)^2
grps$sd_average2 <- sqrt(total / length(grps$group2))

diag(grps$sd_average1) <- NA
diag(grps$sd_average2) <- NA


# Plot double corrplot (limits = max takes the maximum absolute value as the limits for the colors)
grps$double_corrplot <- doublecorrplot(grps$average1, grps$average2, limits = "max", labels = var_labels)
grps$double_corrplot
ggsave(grps$double_corrplot, file = "graphs/double_corrplot.png", units = "cm", width = 16, height = 16*0.9)
ggsave(grps$double_corrplot, file = "graphs/double_corrplot.tiff", units = "cm", width = 16, height = 16*0.9)

# Black and white version
grps$bw_double_corrplot <- doublecorrplot_bw(grps$average1, grps$average2, "max", labels = var_labels)
grps$bw_double_corrplot
ggsave(grps$bw_double_corrplot, file = "graphs/double_corrplot_bw.png", units = "cm", width = 16, height = 16*0.9)
ggsave(grps$bw_double_corrplot, file = "graphs/double_corrplot_bw.tiff", units = "cm", width = 16, height = 16*0.9)




#### FA ----
# Factor analysis
temp1 <- grps$average1
temp2 <- grps$average2
diag(temp1) <- diag(temp2) <- 1
fa_group1 <- psych::fa(temp1, nfactors = 3)
fa_group2 <- psych::fa(temp2, nfactors = 2)
fa_group1
fa_group2

# Predict factor scores, including for those with missing values
df_subset <- df %>%
  dplyr::select(lrscale:anti_libertarianism)

temp1 <- predict(fa_group1, df_subset, missing = TRUE)
colnames(temp1) <- c("cultural1", "economic1")

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



### Plot the network of belief systems (Figure 1)
# Set coords (only the first time, so all networks are comparable)
#grps$coords <- layout.fruchterman.reingold(grps$network)
#V(grps$network)$color <- c("#FFFFFFFF", "#7C7B78FF")[V(grps$network)$community]
#plot(grps$network, layout = grps$coords)

# Remove edges < .40 percentile -> trimmed network
grps$network_trimmed <- grps$network
grps$network_trimmed <- igraph::delete.edges(grps$network_trimmed, 
                                            which(E(grps$network_trimmed)$weight < quantile(E(grps$network_trimmed)$weight, 0)))

# Set colors and transparancy
E(grps$network_trimmed)$color <- rgb(.7,.7,.7, percent_rank(E(grps$network_trimmed)$weight))
V(grps$network_trimmed)$color <- c("#FFFFFFFF", "#7C7B78FF")[V(grps$network)$community]
V(grps$network_trimmed)$label.color <- "black"
plot(grps$network_trimmed, layout = grps$coords)

# Size of node determined by density
V(grps$network_trimmed)$size <- as.numeric(as.vector(class_descriptives[,5]*120))

# Text size
V(grps$network_trimmed)$label.cex <- 3

# Define new object shape, so we can increase border size of vertices
mycircle <- function(coords, v=NULL, params) {
  vertex.color <- params("vertex", "color")
  if (length(vertex.color) != 1 && !is.null(v)) {
    vertex.color <- vertex.color[v]
  }
  vertex.size  <- 1/200 * params("vertex", "size")
  if (length(vertex.size) != 1 && !is.null(v)) {
    vertex.size <- vertex.size[v]
  }
  vertex.frame.color <- params("vertex", "frame.color")
  if (length(vertex.frame.color) != 1 && !is.null(v)) {
    vertex.frame.color <- vertex.frame.color[v]
  }
  vertex.frame.width <- params("vertex", "frame.width")
  if (length(vertex.frame.width) != 1 && !is.null(v)) {
    vertex.frame.width <- vertex.frame.width[v]
  }
  
  mapply(coords[,1], coords[,2], vertex.color, vertex.frame.color,
         vertex.size, vertex.frame.width,
         FUN=function(x, y, bg, fg, size, lwd) {
           symbols(x=x, y=y, bg=bg, fg=fg, lwd=lwd,
                   circles=size, add=TRUE, inches=FALSE)
         })
}

add.vertex.shape("fcircle", clip=igraph.shape.noclip,
                 plot=mycircle, parameters=list(vertex.frame.color=1,
                                                vertex.frame.width=1))

# Set vertex shape etc
V(grps$network_trimmed)$shape = "fcircle"
V(grps$network_trimmed)$frame.color = "black"
V(grps$network_trimmed)$frame.width = 2
