### Demographic differences between groups ----
# Prepare dataset with only countries that have belief systems in both groups
temp <- df %>%
  group_by(country) %>%
  summarize(mean = mean(dummy_community1, na.rm = T))
temp$country[which(temp$mean %% 1 != 0)]

df_subset <- df %>%
  filter(country %in% temp$country[which(temp$mean %% 1 != 0)]) %>%
  select(country, dummy_community1, education, hhincome, female, age, religious, urbanization, ethnic_minority) %>%
  drop_na() 

# Multilevel logistic/binomial regression
mllogreg <- glmer(dummy_community1 ~ education + hhincome + female + scale(age) + religious + urbanization + ethnic_minority + (1 | country),
                  data = df_subset, 
                  family = binomial)
summary(mllogreg)


### Demographic differences between belief systems within countries ----
# Some countries have no measured income
temp <- df %>%
  group_by(class_id) %>%
  summarize(mean = mean(hhincome, na.rm = T))
temp <- c(1:total_classes)[is.na(temp$mean)]

result <- vector(mode = "list", length = total_classes)
a <- 0
sink(file="output.txt") # output to file
for (i in seq_along(countries)) {
  # Make subset of country data and add class membership as dummies to it
  df_subset <- df %>% 
    filter(country == countries[[i]]) %>% 
    mutate(dummy = 1) %>%
    pivot_wider(names_from = cca_membership,
                values_from = dummy,
                values_fill = 0,
                names_prefix = "class")
  
  # Loop through all the classes within country
  for (ii in seq_along(country_classes[[i]])) {
    a <- a + 1
    if (a %in% temp) next
    model <- glm(paste0("class", ii, 
                        " ~ education + hhincome + female + age + religious + urbanization + ethnic_minority"), 
                 data = df_subset, family = "binomial") 
    print(a)
    print(summary(model))
    
    result[[a]] <- formatModel.glm(model)
  }
}
sink()
names(result) <- class_names
result <- result[c(1:total_classes)[-temp]]


### Summary of results and calculate share significance
a <- 1
for (i in seq_len(total_classes)) {
  if (i %in% temp) next
  if (i == 1) summary <- as.data.frame(cbind(class = class_names[i], 
                                             group = grps$group_membership[[i]], 
                                             result[[a]][c(2:9),]))
  else summary <- rbind(summary, cbind(class = class_names[i], 
                                       group = grps$group_membership[[i]], 
                                       result[[a]][c(2:9),]))
  a <- a + 1
}

share_sig <- summary %>%
  mutate(sig = ifelse(p < 0.05, 1, 0)) %>%
  group_by(var) %>%
  summarize(share_sig = mean(sig))
share_sig


#### Calculate partial R2 for all vars over all belief systems ----
vars <- c("education2", "education3", "hhincome", "female", "age", "religious", "urbanization", "ethnic_minority")

# use this line below to check partial r2 of edu_cat variable (overwrites old results)
#vars <- c("edu_cat", "hhincome", "female", "age", "religious", "urbanization", "ethnic_minority")

a <- 0
for (i in seq_along(countries)) {
  # Make subset of country data and add class membership as dummies to it
  df_subset <- df %>% 
    filter(country == countries[[i]]) %>% 
    select(id, country, cca_membership, education, hhincome, female, age, religious, urbanization, ethnic_minority) %>% 
    drop_na() %>%
    mutate(dummy = 1) %>%
    pivot_wider(names_from = cca_membership,
                values_from = dummy,
                values_fill = 0,
                names_prefix = "class") %>%
    mutate(dummy = 1, education_dummy = as.numeric(education)) %>%
    pivot_wider(names_from = education_dummy,
                values_from = dummy,
                values_fill = 0,
                names_prefix = "education")
  
  # Loop through all the classes within country
  for (ii in seq_along(country_classes[[i]])) {
    a <- a + 1
    if (a %in% temp) next
    f <- paste0("class", ii, " ~ ", paste(vars, collapse = " + "))
    r2_full <- summary(lm(f, 
                          data = df_subset))[["r.squared"]]
    res <- data.frame(matrix(nrow = length(vars), ncol = 3))
    for (iii in c(1:length(vars))) {
      f <- paste0("class", ii, " ~ ", paste(vars[-which(vars %in% vars[iii])], collapse = " + "))
      model <- lm(f, 
                  data = df_subset) 
      res[iii,] <- c(class_names[ii], vars[iii], r2_full - summary(model)[["r.squared"]])
    }
    res <- cbind(res, r2_full)
    if (a == 1) {
      R2_chk_regs <- res
      a <- 2
    } else R2_chk_regs <- rbind(R2_chk_regs, res)
  }
}
partial_r2 <- R2_chk_regs %>% 
  mutate(X2 = factor(X2, levels = unique(R2_chk_regs$X2))) %>% 
  group_by(X2) %>% 
  summarise(meanPR2 = mean(as.numeric(X3)))
partial_r2