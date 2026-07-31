# Factor analyses across belief systems ----
# Two-factor solution
bs_factoranalysis <- function(class_name) {
  # performs fa per country, extracts two first factors, returns variance accounted table and loadings
  cormatrix <- classes[[class_name]]
  diag(cormatrix) <- 1
  fa <- psych::fa(cormatrix, nfactors = 2)
  
  MR1 <- fa$loadings[, 1]
  MR2 <- fa$loadings[, 2]
  
  return(list(fa$Vaccounted,
              with(fa, as.data.frame(loadings[1:nrow(loadings), 1:ncol(loadings)])),
              fa$Phi[1,2]))
}
fa_classes <- map(class_names, bs_factoranalysis) |> 
  set_names(class_names)

## Data frame with all factors and info ----
bs_fa_summary <- map2(
  fa_classes, names(fa_classes),
  \(fa_result, bs_name) {
    corr <- fa_result[[3]]
    fa_result <- list(fa_result[[1]], fa_result[[2]])
    rownames(fa_result[[1]]) <- c("ss_loadings", "var_explained", "cum_var", "prop_explained", "cum_prop")
    as_tibble(t(list_rbind(map(fa_result, data.frame)))) |> 
      mutate(
        bs = bs_name,
        corr,
        dimension = c("MR1", "MR2"),
        group = if_else(bs_name %in% grps$group1, 1, 2),
        count = rowSums((across(lrscale:anti_libertarianism, \(x) (x > 0.3)))),
        .before = 1
      ) |> select(-cum_var, -prop_explained, -cum_prop)
  }
) |> list_rbind()

# One- or two-dimensionality ----
## How many factors/components with eigen value > 1? ----
map_vec(
  class_names, 
  \(x) {
    cormatrix <- classes[[x]]
    diag(cormatrix) <- 1
    fa <- psych::fa(cormatrix)
    return(sum(fa$e.values > 1))
  })

## Patterns in variance explained ----
### One factor solution ----
fa1_classes <- map(class_names, \(x) {
  cormatrix <- classes[[x]]
  diag(cormatrix) <- 1
  fa <- psych::fa(cormatrix, nfactors = 1)
  
  return(fa)
})
temp <- map_vec(fa1_classes, \(fa) fa$Vaccounted[2,1])
psych::describe(temp) |> as.data.frame() |> round(3)
psych::describe(temp[which(class_names %in% grps$group1)]) |> as.data.frame() |> round(3)
psych::describe(temp[which(class_names %in% grps$group2)]) |> as.data.frame() |> round(3)

### Two factor solution ----
temp <- bs_fa_summary |> 
  select(bs:group, var_explained) |> 
  pivot_wider(names_from = dimension, values_from = var_explained) |> 
  mutate(sum = MR1 + MR2,
         diff = MR1 - MR2) |> 
  arrange(diff)

temp |> filter(group == 1) |> print(n = 45973)
temp |> filter(group == 2) |> print(n = 45973)

temp |> select(-bs) |> psych::describe(fast = TRUE) |> as.data.frame() |> round(3)
temp |> filter(group == 1) |> select(-bs) |> psych::describe(fast = TRUE) |> as.data.frame() |> round(3)
temp |> filter(group == 2) |> select(-bs) |> psych::describe(fast = TRUE) |> as.data.frame() |> round(3)


## How many variables load on factors? ----
### One-factor solution 
temp <- map_vec(
  fa1_classes, 
  \(fa) {
    loadings <- fa$loadings[,1]
    sum(ceiling(loadings[which(loadings > 0.3)]))
  }
)
psych::describe(temp) |> as.data.frame() |> round(3)
psych::describe(temp[which(class_names %in% grps$group1)]) |> as.data.frame() |> round(3)
psych::describe(temp[which(class_names %in% grps$group2)]) |> as.data.frame() |> round(3)

### Two-factor solution
# Data frame with descriptive statistics of the number of variables that load > 0.3 on each factor
bs_fa_summary |> reframe(
  m = mean(count),
  sd = sd(count),
  min = min(count),
  max = max(count),
  .by = c(dimension)
)




# What are the dimensions? ----
## Count number of loadings over 0.3 that pairs of beliefs have on the same factor ----
# Group 1
common_loadings1 <- bs_fa_summary |> 
  filter(group == 1) |> 
  select(lrscale:anti_libertarianism)
common_loadings1 <- ceiling(abs(common_loadings1)-0.3)
common_loadings1 <- t(as.matrix(common_loadings1)) %*% as.matrix(common_loadings1)

common_loadings1[lower.tri(common_loadings1, diag = TRUE)] <- NA
common_loadings1 <- common_loadings1 |> 
  as_tibble(rownames = "var1") |> 
  pivot_longer(cols = -var1, names_to = "var2", values_to = "count")

# Group 2
common_loadings2 <- bs_fa_summary |> 
  filter(group == 2) |> 
  select(lrscale:anti_libertarianism)
common_loadings2 <- ceiling(abs(common_loadings2)-0.3)
common_loadings2 <- t(as.matrix(common_loadings2)) %*% as.matrix(common_loadings2)

common_loadings2[lower.tri(common_loadings2, diag = TRUE)] <- NA
common_loadings2 <- common_loadings2 |> 
  as_tibble(rownames = "var1") |> 
  pivot_longer(cols = -var1, names_to = "var2", values_to = "count")

# Print 20 most often co-occuring belief-pairs
common_loadings1 |> slice_max(count, n = 20) |> print(n = 20)
common_loadings2 |> slice_max(count, n = 20) |> print(n = 20)



## Classify factors based on mean loading ----
bs_fa_classified <- bs_fa_summary |> 
  select(bs, dimension, group, 
         anti_lgbt, anti_immigration, 
         anti_egalitarianism, no_resp_gov) |> 
  mutate(
    cultural = (anti_lgbt + anti_immigration)/2,
    economic = (anti_egalitarianism + no_resp_gov)/2,
    cultdim = case_when(
      abs(cultural) < 0.3 ~ 0,
      .default = sign(cultural)
    ),
    econdim = case_when(
      abs(economic) < 0.3 ~ 0,
      .default = sign(economic)
    ),
    classified_as = case_when(
      abs(cultural) > 0.3 & abs(economic) > 0.3 ~ "both",
      abs(cultural) > 0.3 ~ "cultural",
      abs(economic) > 0.3 ~ "economic",
      .default = "none"
    ))

### Graph ----
lim <- 0.66
ggplot(
  bs_fa_classified, 
  aes(
    cultural, 
    economic, 
    fill = factor(group), 
    shape = dimension
  )
) +
  geom_point(color = "Black") +
  geom_vline(xintercept = c(-0.3, 0.3), linewidth = 0.3) + 
  geom_hline(yintercept = c(-0.3, 0.3), linewidth = 0.3) +
  scale_x_continuous(limits = c(-lim, lim), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-lim, lim), expand = c(0, 0)) +
  scale_fill_manual(values = c("#FFFFFFFF", "#7C7B78FF")) +
  guides(fill = guide_legend(override.aes=list(shape = 21))) +
  scale_shape_manual(values = c("MR1" = 21, "MR2" = 24)) +
  labs(
    x = "Mean cultural loading", 
    y = "Mean economic loading", 
    fill = "Group", 
    shape = "Dimension"
  )
ggsave("graphs/bsfa_cultecon.png", units = "cm", width = 16, height = 8)


with(bs_fa_classified, table(econdim, cultdim))
with(bs_fa_classified, table(econdim, cultdim, group))
with(bs_fa_classified, table(econdim, cultdim, dimension))
with(bs_fa_classified, table(econdim, cultdim, dimension, group))

### Check across all belief systems whether they contain any cultural / economic dimension
bs_fa_classified |> 
  select(bs, dimension, cultdim, econdim) |> 
  pivot_wider(names_from = dimension, values_from = c(cultdim, econdim)) |> 
  mutate(any = abs(cultdim_MR1) + abs(cultdim_MR2) + abs(econdim_MR1) + abs(econdim_MR2)) |> 
  print(n = 48573)


#### Group 2 cultural with anti_migration and welfare_chauvinism
temp <- bs_fa_summary |> 
  select(bs, corr, dimension, group, ss_loadings, anti_lgbt, anti_immigration, welfare_chauvinism) |> 
  mutate(
    cultural1 = (anti_lgbt + anti_immigration)/2,
    cultural2 = (anti_immigration + welfare_chauvinism)/2,
    cultdim1 = case_when(
      cultural1 > 0.3 ~ 1,
      cultural1 < 0.3 & cultural1 > -0.3 ~ 0,
      cultural1 < -0.3 ~ -1
    ),
    cultdim2 = case_when(
      cultural2 > 0.3 ~ 1,
      cultural2 < 0.3 & cultural2 > -0.3 ~ 0,
      cultural2 < -0.3 ~ -1
    ))
with(temp, table(cultdim1, cultdim2, group))


## Correlation between ordinary cultural & economic scales ---- 
df_subset <- df |> 
  select(-c(cultural1:cultural2)) |> 
  drop_na(class_community)
df_subset$cultural <- rowMeans(select(df_subset, "anti_lgbt", "anti_immigration"))
df_subset$economic <- rowMeans(select(df_subset, "anti_interventionism", "anti_egalitarianism"))

temp <- map_vec(
  unique(df_subset$class_name), 
  \(x) {
    with(df_subset |> filter(class_name == x), cor(cultural, economic, use = "complete.obs"))
  }) |> set_names(unique(df_subset$class_name))

psych::describe(temp[which(names(temp) %in% grps$group1)])
psych::describe(temp[which(names(temp) %in% grps$group2)])


