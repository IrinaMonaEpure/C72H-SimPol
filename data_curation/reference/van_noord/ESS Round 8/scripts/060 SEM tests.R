produce_syntax <- function(x) {
  list <- list()
  len <- length(x)
  for (i in 1:len) {
    j <- i + 1
    list[[i]] <- paste(x[i], 
                       "~~", 
                       paste(x[j:len], collapse = " + "))
  }
  paste(list, collapse = "||")
}
produce_syntax(names(df_subset[,1:20])) # adjust in notepad so that its correct, but the below is what you should get:

model <- '
lrscale               ~~ gender_inequality + anti_lgbt + euroscepticism + anti_immigration + anti_egalitarianism + benefits_eco + benefits_soc + welfare_chauvinism + anti_interventionism + anti_sb_lowincome + anti_sb_parents + spend_edu + anti_basinc + anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
gender_inequality     ~~ anti_lgbt + euroscepticism + anti_immigration + anti_egalitarianism + benefits_eco + benefits_soc + welfare_chauvinism + anti_interventionism + anti_sb_lowincome + anti_sb_parents + spend_edu + anti_basinc + anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
anti_lgbt             ~~ euroscepticism + anti_immigration + anti_egalitarianism + benefits_eco + benefits_soc + welfare_chauvinism + anti_interventionism + anti_sb_lowincome + anti_sb_parents + spend_edu + anti_basinc + anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
euroscepticism        ~~ anti_immigration + anti_egalitarianism + benefits_eco + benefits_soc + welfare_chauvinism + anti_interventionism + anti_sb_lowincome + anti_sb_parents + spend_edu + anti_basinc + anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
anti_immigration      ~~ anti_egalitarianism + benefits_eco + benefits_soc + welfare_chauvinism + anti_interventionism + anti_sb_lowincome + anti_sb_parents + spend_edu + anti_basinc + anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
anti_egalitarianism   ~~ benefits_eco + benefits_soc + welfare_chauvinism + anti_interventionism + anti_sb_lowincome + anti_sb_parents + spend_edu + anti_basinc + anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
benefits_eco          ~~ benefits_soc + welfare_chauvinism + anti_interventionism + anti_sb_lowincome + anti_sb_parents + spend_edu + anti_basinc + anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
benefits_soc          ~~ welfare_chauvinism + anti_interventionism + anti_sb_lowincome + anti_sb_parents + spend_edu + anti_basinc + anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
welfare_chauvinism    ~~ anti_interventionism + anti_sb_lowincome + anti_sb_parents + spend_edu + anti_basinc + anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
anti_interventionism  ~~ anti_sb_lowincome + anti_sb_parents + spend_edu + anti_basinc + anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
anti_sb_lowincome     ~~ anti_sb_parents + spend_edu + anti_basinc + anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
anti_sb_parents       ~~ spend_edu + anti_basinc + anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
spend_edu             ~~ anti_basinc + anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
anti_basinc           ~~ anti_cli_taxes + anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
anti_cli_taxes        ~~ anti_cli_renewable + anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
anti_cli_renewable    ~~ anti_cli_ban + cli_sceptic + authoritarianism + anti_libertarianism
anti_cli_ban          ~~ cli_sceptic + authoritarianism + anti_libertarianism
cli_sceptic           ~~ authoritarianism + anti_libertarianism
authoritarianism      ~~ anti_libertarianism
'

# Function to fit two models for a given country
do_sem <- function (country_name, data = df) {
  df_subset <- data %>%
    filter(country == country_name & !is.na(cca_membership)) %>%
    select(lrscale:anti_libertarianism, cca_membership)
  
  fit_single <- sem(model, data = df_subset, missing = "ml", 
                    group = "cca_membership", 
                    group.equal = c("loadings", "composite.loadings", "intercepts", "means", "thresholds", 
                                    "regressions", "residuals", "residual.covariances", "lv.variances", "lv.covariances"))
  
  print(paste("country:", country_name, "single fitted"))
  
  fit_groups <- sem(model, data = df_subset, missing = "ml", 
                    group = "cca_membership")
  print(paste("country:", country_name, "groups fitted"))
  
  return(list(fit_single,
              fit_groups))
}

# Perform function for each contry
SEM_results <- map(unique(df$country), do_sem)
names(SEM_results) <- unique(df$country)

# Function to extract fit values (AIC, BIC, Chi.sq)
get_fits <- function (x) {
  test <- lavTestLRT(x[[1]], x[[2]])
  
  data.frame(single_aic = test$AIC[2],
             groups_aic = test$AIC[1],
             aic_diff   = test$AIC[1]-test$AIC[2],
             single_bic = test$BIC[2],
             groups_bic = test$BIC[1],
             bic_diff   = test$BIC[1]-test$BIC[2],
             Xdiff      = test$`Chisq diff`[2],
             DFdiff     = test$`Df diff`[2],
             Xp         = test$`Pr(>Chisq)`[2])
}
# Perform function for each country
SEM_summary <- map_dfr(SEM_results, get_fits) 

# Save SEM_results to file and delete file to save memory
save(SEM_results, file = "data/SEM_results.RData")
rm(SEM_results)

# Calculate difference, negative values indicate groups fit is better than single matrix fit
SEM_summary %>%
  mutate(country = unique(df$country), .before = 1,
         Xp = round(Xp, digits = 4))
  





df_subset <- df %>%
  filter(!is.na(class_community)) %>%
  select(lrscale:anti_libertarianism, class_community)

fit_single <- sem(model, data = df_subset, missing = "ml", 
                  group = "class_community", 
                  group.equal = c("loadings", "composite.loadings", "intercepts", "means", "thresholds", 
                                  "regressions", "residuals", "residual.covariances", "lv.variances", "lv.covariances"))
summary(fit_single, fit.measures = T, standardized = T)

fit_groups <- sem(model, data = df_subset, missing = "ml", 
                  group = "class_community")
summary(fit_groups, fit.measures = T, standardized = T)

lavTestLRT(fit_single, fit_groups)




cor(df_subset[,1:20], use = "complete.obs")

get_fits(SEM_results$NL)
lavTestLRT(SEM_results$AT[[1]], SEM_results$AT[[2]])




#### ----
# Function to fit two models for a given country
do_sem <- function (data) {
  fit_single <- sem(model, data = data, missing = "ml", 
                    group = "cca_membership", 
                    group.equal = c("loadings", "composite.loadings", "intercepts", "means", "thresholds", 
                                    "regressions", "residuals", "residual.covariances", "lv.variances", "lv.covariances"))
  
  fit_groups <- sem(model, data = data, missing = "ml", 
                    group = "cca_membership")
  
  return(list(fit_single,
              fit_groups))
}

df_subset <- df %>%
  filter(country == "PL" & !is.na(cca_membership)) %>%
  select(lrscale:anti_libertarianism, cca_membership)

temp <- do_sem(df_subset %>% filter(cca_membership %in% c(1,2)))
get_fits(temp)

temp <- do_sem(df_subset %>% filter(cca_membership %in% c(1,3)))
get_fits(temp)

temp <- do_sem(df_subset %>% filter(cca_membership %in% c(1,4)))
get_fits(temp)

temp <- do_sem(df_subset %>% filter(cca_membership %in% c(2,3)))
get_fits(temp)

temp <- do_sem(df_subset %>% filter(cca_membership %in% c(2,4)))
get_fits(temp)

temp <- do_sem(df_subset %>% filter(cca_membership %in% c(3,4)))
get_fits(temp)

summary(temp[[1]], fit.measures = T, standardized = T)
summary(temp[[2]], fit.measures = T, standardized = T)



