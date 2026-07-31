# Packages needed
library(plyr)
library(tidyverse)
library(rio)
library(doParallel)
#library(corclass) # adjusted algorithm, so not necessary, but this is the original package
library(corrplot)
library(qgraph)
library(igraph)
library(writexl)
library(lme4)
library(lmerTest)
library(ggeffects)

# Packages needed elsewhere, but used with :: so no need to load
#library(scales)
#library(ggforce)
#library(broom)
#library(broom.mixed)
#library(evolqg)


# Import dataset
df = import('data/df_ESS4.RData')

### CCA analysis ----
countries <- unique(df$country)
countries

# Load bscca results from file
# Don't run normal CCA after this (or put in different list)
countries_cca = import('data/countries_cca.RData')

# List of results
countries_cca <- vector("list", length(countries))

# Perform CCA with adjusted algorithm, in parallel mode so runs faster
# On pretty fast PC it's about 3-5 minutes per country. All countries on one core could thus take a couple of hours
# If replicating, check how many cores your processor has, to predict how long it takes to run all countries
# And/or perform bscca on only one country to benchmark time per country
cl <- makeCluster(detectCores() - 1)
system.time({
  registerDoParallel(cl)
  countries_cca <- foreach(i = seq_along(countries), .packages=c("tidyverse")) %dopar% {
    df_subset <- df %>% 
      filter(country == countries[[i]]) %>% # Filter by country
      select(lrscale:anti_libertarianism) # Select only beliefs
    countries_cca[i] <- bscca(df_subset, zero.action = "ownclass") # Perform modified CCA
  }})
stopCluster(cl)

# Save to file
save(countries_cca, file = "data/countries_cca.RData")

### Clean results ----
# Remove degenerate classes and reorder membership data
# cca will probably still say that there is one more class, but it will only contain the non-degenerate classes
for (i in c(1:length(countries))) {
  degenerate <- 0
  for (ii in 1:length(countries_cca[[i]]$modules)) {
    k <- ii - degenerate
    if (countries_cca[[i]]$modules[[k]]$degenerate == T) { 
      # Class is degenerate so shift membership data one place
      countries_cca[[i]]$membership <- if_else(countries_cca[[i]]$membership == k, 
                                               NA_real_, countries_cca[[i]]$membership)
      countries_cca[[i]]$membership <- if_else(countries_cca[[i]]$membership > k, 
                                               countries_cca[[i]]$membership - 1, countries_cca[[i]]$membership)
      countries_cca[[i]]$modules[[k]] <- NULL
      degenerate <- degenerate + 1
    }
  }
}
rm(degenerate)

# Move modules into a list of list of classes
classes <- list()
country_membership <- list()
class_membership <- vector()
class_names <- vector()
country_classes <- list()
k <- 1
for (i in c(1:length(countries))) {
  country_classes_temp <- vector()
  country_membership[[i]] <- countries_cca[[i]]$membership
  for (ii in 1:length(countries_cca[[i]]$modules)) {
    classes[[k]] <- countries_cca[[i]]$modules[[ii]]$cormat # save class to list
    diag(classes[[k]]) <- NA
    class_membership <- c(class_membership, table(countries_cca[[i]]$membership)[ii]) # save membership to vector
    class_names <- c(class_names, paste0(countries[[i]], "_", ii)) # save class names: country_class#
    country_classes_temp <- c(country_classes_temp, paste0(countries[[i]], "_", ii))
    k <- k + 1
  }
  country_classes[[i]] <- country_classes_temp
  rm(country_classes_temp)
}

names(class_membership) <- class_names
names(classes) <- class_names
names(country_classes) <- countries
names(country_membership) <- countries

total_classes <- length(classes) # total number of classes
total_countries <- length(countries) # total number of countries
var_names <- colnames(classes[[1]]) # all names of variables
total_vars <- length(var_names) # total number of variables
var_labels <- c("Left-right identification",
                "Gender inequality",
                "Anti-LGBT",
                "Euroscepticism",
                "Anti-immigration",
                "Anti-egalitarianism",
                "Benefits harm economy",
                "Benefits harm society",
                "Welfare chauvinism",
                "Anti-interventionism",
                "Harsh sentences",
                "Anti-militant democracy",
                "No science-environment solution",
                "Anti-government spending",
                "Regressive taxes",
                "Regressive benefits",
                "Age prejudice",
                "Authoritarianism",
                "Anti-libertarianism") # Labels for the beliefs

### Add membership to dataframe ----
# Add CCA membership data to df
df <- df %>%
  mutate(cca_membership = unlist(country_membership))

# Add a class id variable (separately for those with missing membership (e.g. degenerate class members), so they get NA rather than a number)
temp <- df %>%
  filter(is.na(cca_membership)) %>%
  mutate(class_id = NA)
  
df <- df %>%
  filter(!is.na(cca_membership)) %>%
  group_by(country, cca_membership) %>%
  mutate(class_id = cur_group_id()) %>%
  ungroup()

df <- rbind(df, temp) %>%
  arrange(id)


### Explanation for the list of classes ----
# classes is a single list with all classes, can be accessed with classes[n] where n is the number of the class, but also via classes["AT_1"] where the string is the country code + the number of the class in that country. 
# 
# This can also be done with multiple classes with a vector: classes[c("AT_1", "DE_1")]. 
# 
# country_classes can give a list of classes (or names of these) for each country. So country_classes[[1]] gives a list of classes for the first country. classes[country_classes[[1]]] gives the actual correlation matrices for that country. Note the single brackets with classes, and double brackets with country_classes!
#
# Instead of classes you can use the class_membership vector to get the n of each class, e.g. class_membership[[1]]
# country_membership gives the raw membership data of all respondents per country, e.g. country_membership[[1]]
#
# class_names is just a vector of all class_names, var_names is a vector of all variable names, countries is a vector with all country names