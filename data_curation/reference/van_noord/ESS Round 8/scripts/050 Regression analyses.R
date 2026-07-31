### Demographic differences between groups ----
# Prepare dataset with only countries that have belief systems in both groups
df_subset <- df %>%
  select(id, country, dummy_community1, education, hhincome, female, age, religious, urbanization, ethnic_minority) %>%
  filter(country %in% c("AT", "BE", "CH", "DE", "EE", "FR", "GB", "IE", "IL", "IT", "PL", "SI")) %>%
  drop_na() %>%
  mutate(female = as.numeric(female),
         religious = as.numeric(religious),
         ethnic_minority = as.numeric(ethnic_minority),
         hhincome = scale(hhincome),
         age = scale(age),
         urbanization = scale(urbanization)) # mutate predictors to numeric as otherwise the predicted mean is implausibly low compared to raw data (0.35 vs. 0.43) with these data predicted mean is closer to raw data (0.41 vs. 0.43). Coefficients of predicted are unaffected.

# Multilevel logistic/binomial regression
mllogreg <- glmer(dummy_community1 ~ education + hhincome + female + age + religious + urbanization + ethnic_minority + (1 | country),
                  data = df_subset, 
                  family = binomial)
summary(mllogreg)

# Predict values
res <- rbind(
  cbind("var" = "Income", as.data.frame(ggemmeans(mllogreg, terms = c("hhincome [minmax]")))),
  cbind("var" = "Income", as.data.frame(ggemmeans(mllogreg, terms = c("hhincome [meansd]")))),
  cbind("var" = "Education", as.data.frame(ggemmeans(mllogreg, terms = c("education [all]")))),
  cbind("var" = "Gender", as.data.frame(ggemmeans(mllogreg, terms = c("female [all]")))),
  cbind("var" = "Age", as.data.frame(ggemmeans(mllogreg, terms = c("age [minmax]")))),
  cbind("var" = "Age", as.data.frame(ggemmeans(mllogreg, terms = c("age [meansd]")))),
  cbind("var" = "Religious", as.data.frame(ggemmeans(mllogreg, terms = c("religious [all]")))),
  cbind("var" = "Urbanization", as.data.frame(ggemmeans(mllogreg, terms = c("urbanization [minmax]")))),
  cbind("var" = "Urbanization", as.data.frame(ggemmeans(mllogreg, terms = c("urbanization [meansd]")))),
  cbind("var" = "Ethnic minority", as.data.frame(ggemmeans(mllogreg, terms = c("ethnic_minority [all]")))))

res[4,3] # Overall mean

# Label data correctly for graph
res$labels <- c("Min", "Max", "-1SD", "Mean", "+1SD", 
                "Low", "Middle", "High", 
                "Male", "Female", 
                "Min", "Max", "-1SD", "Mean", "+1SD", 
                "No", "Yes", 
                "Min", "Max", "-1SD", "Mean", "+1SD", 
                "No", "Yes")
res$var <- factor(res$var, levels = c("Education", "Income", "Gender", "Age", "Religious", "Urbanization", "Ethnic minority"))
res

# Calculate predicted scores for Group 2 membership
temp1 <- res[,c(1:3,7,8)] %>%
  mutate(group = as.numeric(group))
temp2 <- temp1 %>%
  mutate(group = 2,
         predicted = 1-predicted)

# Add together in one df
temp <- bind_rows(temp1, temp2) %>%
  mutate(group = factor(group)) %>%
  unite("name", var, labels, remove = FALSE) %>%
  mutate(group = factor(group, labels = c("Group 1", "Group 2")),
         labels = factor(labels, levels = c("Max", "+1SD", "Mean", "-1SD", "Min", "High", "Middle", "Low", "Female", "Male", "Yes", "No")))

# Plot stacked bar charts
ggplot(data = temp, aes(x = predicted, y = labels)) +
  geom_col(aes(fill = group), width = 0.8, position = position_stack(reverse = TRUE), color = "black", size = 0.4) +
  geom_vline(xintercept = temp[4,4], color = "#00000099") +
  facet_grid(rows = vars(var), scales = "free", space = "free") +
  theme(strip.text.y = element_text(angle = 0), panel.spacing.y = unit(0.7, "line"),
        panel.background = element_rect(fill = "white", colour = "white"),
        legend.position = "bottom") +
  scale_x_continuous(expand = c(0,0), breaks = c(0:10)/10) + 
  scale_y_discrete(expand = c(0,0)) + 
  scale_fill_manual(values = c("#FFFFFFFF", "#7C7B78FF")) +
  labs(x = "Predicted group membership", y = "", fill = "Group") 
ggsave(file = "graphs/demogr_predicts.png", units = "cm", width = 16, height = 16*0.75)
ggsave(file = "graphs/demogr_predicts.tiff", units = "cm", width = 16, height = 16*0.75)


### Demographic differences between belief systems within countries ----
result <- vector(mode = "list", length = total_classes)
a <- 1
sink(file="output.txt") # output to file
for (i in seq_along(countries)) {
  # Make subset of country data and add class membership as dummies to it
  df_subset <- df %>% 
    filter(country == countries[[i]]) %>% 
    mutate(dummy = 1) %>%
    pivot_wider(names_from = cca_membership,
                values_from = dummy,
                values_fill = 0,
                names_prefix = "class") %>%
    mutate_at(vars(hhincome, age, urbanization), 
              scales::rescale, to = c(0,1)) # Rescale continuous vars to 0-1 so effect size is somewhat comparable
  
  # Loop through all the classes within country
  for (ii in seq_along(country_classes[[i]])) {
    model <- glm(paste0("class", ii, 
                         " ~ education + hhincome + female + age + religious + urbanization + ethnic_minority"), 
                  data = df_subset, family = "binomial") 
    print(a)
    print(summary(model))
    
    temp <- formatModel.glm(model)
    result[[a]] <- temp
    a <- a + 1
  }
}
sink()
names(result) <- class_names

# Save to file
write_xlsx(result, 
           path = "tables/binomial_results.xlsx")


### Summary of results and calculate share significance
for (i in seq_along(classes)) {
  if (i == 1) summary <- as.data.frame(cbind(class = class_names[i], 
                                             group = grps$group_membership[[i]], 
                                             result[[i]][c(2:9),]))
  else summary <- rbind(summary, cbind(class = class_names[i], 
                                       group = grps$group_membership[[i]], 
                                       result[[i]][c(2:9),]))
}

share_sig <- summary %>%
  mutate(sig = ifelse(p < 0.05, 1, 0)) %>%
  group_by(var) %>%
  summarize(share_sig = mean(sig))
share_sig


#### Calculate partial R2 for all vars over all belief systems ----
vars <- c("education2", "education3", "hhincome", "female", "age", "religious", "urbanization", "ethnic_minority")

# use this line below to check partial r2 of edu_cat variable (overwrites old results)
#vars <- c("education", "hhincome", "female", "age", "religious", "urbanization", "ethnic_minority")

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
    if (a == 0) {
      R2_chk_regs <- res
      a <- 1
    } else R2_chk_regs <- rbind(R2_chk_regs, res)
  }
}
partial_r2 <- R2_chk_regs %>% 
  mutate(X2 = factor(X2, levels = unique(R2_chk_regs$X2))) %>% 
  group_by(X2) %>% 
  summarise(meanPR2 = mean(as.numeric(X3)))
partial_r2



### Effects of belief system group on vote behavior (in stata) ----
# Prepare data and export to .dta
library(haven)
df_subset <- df %>%
  select(id, country, class_community, vote_behavior, education, hhincome, female, age, religious, urbanization, ethnic_minority) %>%
  filter(country %in% c("AT", "BE", "CH", "DE", "EE", "FR", "GB", "IE", "IL", "IT", "PL", "SI")) %>%
  drop_na()
df_subset <- janitor::clean_names(df_subset)
write_dta(df_subset, file.path(getwd(), "vote_behavior_expanded.dta"))

## Do analysis in Stata 

# Read stata output
library("readxl")
stata_output <- t(read_excel("tables/Stata margins_expanded.xlsx"))[2:15,c(1,5,6)] # Only b, and confidence interval
stata_output <- apply(stata_output, 2, as.numeric)
stata_output <- data.frame(group = rep(c("Group 1", "Group 2"), 7), 
                           vote_type = rep(c("Loyalty", rep("Voice", 5), "Exit"), each = 2),
                           vote = rep(c("Loyalty", "Far left", "Populist \nfar left", "Populist", "Populist \nfar right", "Far right", "Exit"), 
                                      each = 2), 
                           est = stata_output[,1],
                           lb = stata_output[,2],
                           ub = stata_output[,3])
stata_output

# Covert name cols into factor variables
stata_output$group      <- factor(stata_output$group, 
                                  labels = unique(stata_output$group))
stata_output$vote_type  <- factor(stata_output$vote_type, 
                                  levels = unique(stata_output$vote_type), labels = unique(stata_output$vote_type))
stata_output$vote       <- factor(stata_output$vote, 
                                  levels = unique(stata_output$vote), 
                                  labels = unique(stata_output$vote))
stata_output

# Voice total?
with(stata_output, sum(est[which(vote_type == "Voice" & group == "Group 1")]))
with(stata_output, sum(est[which(vote_type == "Voice" & group == "Group 2")]))

# Plot graph
ggplot(stata_output, aes(vote, est, fill = group)) +
  geom_col(position = "dodge", color = "black", size = 0.1) +
  geom_errorbar(aes(ymin = lb, ymax = ub), width = 0.2, size = 0.1, position = position_dodge(width = 0.9)) +
  ylim(-0.015,1) +
  labs(x = "Vote behavior", y = "Predicted probability", fill = "Group") + 
  facet_grid(cols = vars(vote_type), scales = "free", space = "free", switch = "x",
             labeller = labeller(vote_type = c("Loyalty" =  "", "Voice" = "Voice", "Exit" = ""))) +
  theme(panel.spacing.x = unit(0.3, "line"), 
        strip.placement = "outside", 
        strip.background = element_blank(),
        legend.position = "bottom",
        legend.title = element_blank(),
        axis.title.x = element_blank()) +
  scale_fill_manual(values = c("#FFFFFFFF", "#7C7B78FF"))

ggsave("graphs/vote_behavior_mnlogit_extended.png", width = 16, height = 8, units = "cm")
ggsave("graphs/vote_behavior_mnlogit_extended.tiff", width = 16, height = 8, units = "cm")
