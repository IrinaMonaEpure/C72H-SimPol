library(plyr)
library(dplyr)
library(tidyverse)
library(sjmisc)
library(psych)
library(summarytools)
library(essurvey)

##### Import ESS round 4 file ----
set_email("j.van.noord@rug.nl")
df_raw <- import_rounds(4)

##### Select respondents and variables, rename variables ----
df <- df_raw %>%
  filter(agea > 17 | is.na(agea)) %>% # filter out participants under 18
  select(idno, cntry, # General variables
         gndr, rlgblg, blgetmg, domicil, agea, hinctnta, edulvla, # Demographics
         lrscale, freehms, euftf, dfincac, imsclbn, hrshsnt, prtyban, scnsenv, ditxssp, txearn, buproag, # beliefs
         mnrgtjb, wmcpwrk, 
         imsmetn, imdfetn, impcntr, 
         smdfslv, gincdif, 
         sbstrec, sbbsntx, 
         sbprvpv, sbeqsoc, 
         gvslvol, gvslvue, gvcldcr, gvjbevn, gvhlthc, gvpdlwk, 
         impsafe, ipfrule, ipbhprp, ipstrgv, imptrad, 
         impdiff, ipadvnt, ipcrtiv, impfree, ipudrst, 
         earnpen, earnueb) 

df <- df %>%
  rename(id = idno, # General variables
         country = cntry, 
         female = gndr, # Demographics
         religious = rlgblg, 
         ethnic_minority = blgetmg, 
         urbanization = domicil, 
         age = agea, 
         hhincome = hinctnta, 
         education = edulvla, 
         anti_lgbt = freehms, # Single item beliefs (some need reverse coding)
         euroscepticism = euftf, 
         income_meritocracy = dfincac, # does not scale well with other egalitarianism items
         welfare_chauvinism = imsclbn, 
         harsh_sentences = hrshsnt, # scales a little with authoritarianism, but separate to be sure
         anti_mil_democracy = prtyban,
         science_environment = scnsenv,
         government_spending = ditxssp,
         regressive_taxes = txearn, # needs more complicated recoding
         age_prejudice = buproag) 



##### Recoding ----
### Reverse code
reverse_var <- function (x) (min(x, na.rm = T) + max(x, na.rm = T)) - x

df <- df %>%
  mutate(across(c(urbanization, religious, ethnic_minority, # Demographics
                  euroscepticism, income_meritocracy, harsh_sentences, government_spending, age_prejudice, # Single item beliefs
                  gvslvol, gvslvue, gvcldcr, gvjbevn, gvhlthc, gvpdlwk, # Anti-interventionism
                  sbstrec, sbbsntx, # Benefits harms economy
                  impsafe, ipfrule, ipstrgv, ipbhprp, imptrad), # Authoritarianism
                reverse_var)) # Reverse code

### Regressive taxation recoding
table(df$regressive_taxes, useNA = "always")
df <- df %>%
  mutate(regressive_taxes = case_when(
    regressive_taxes == 2 ~ 1,
    regressive_taxes == 1 ~ 2,
    regressive_taxes == 3 ~ 3,
    TRUE ~ NA_real_))
table(df$regressive_taxes, useNA = "always")

# Code 4 as missing in regressive benefits items
df$earnpen[df$earnpen == 4] <- NA
df$earnueb[df$earnueb == 4] <- NA

### Education in three subgroups
table(df$education)
df <- df %>%
  mutate(education = case_when(
    education %in% c("1","2") ~ 1,
    education %in% c("3","4") ~ 2,
    education %in% c("5") ~ 3),
    education = factor(education, levels = c(1, 2, 3), 
                       labels = c("Lower educated", "Middle educated", "Higher educated")))
table(df$education)

### Factorize other variables
df <- df %>%
  mutate(female = factor(female, levels = c(1, 2), 
                         labels = c("Male", "Female")),
         religious = factor(religious, levels = c(1, 2), 
                            labels = c("Non-religious", "Religious")),
         ethnic_minority = factor(ethnic_minority, levels = c(1, 2), 
                                  labels = c("Ethnic majority", "Ethnic minority")))


##### Scales ----
# Gender inequality <- 0.63
psych::alpha(select(df, mnrgtjb, wmcpwrk), check.keys = TRUE)
df$gender_inequality <- with(df, rowMeans(cbind(mnrgtjb, wmcpwrk)))

# Anti-migration .87
psych::alpha(select(df, imsmetn, imdfetn, impcntr))
df$anti_immigration <- rowMeans(select(df, imsmetn, imdfetn, impcntr))
describe(df$anti_immigration)

# Anti-Egalitarianism .60 / r = 0.4247869
psych::alpha(select(df, income_meritocracy, gincdif, smdfslv))
cor(select(df, gincdif, smdfslv), use = "complete.obs")
psych::alpha(select(df, gincdif, smdfslv))
df$anti_egalitarianism <- rowMeans(select(df, gincdif, smdfslv))
describe(df$anti_egalitarianism)

# Anti-interventionism .84
psych::alpha(select(df, gvslvol, gvslvue, gvcldcr, gvjbevn, gvhlthc, gvpdlwk))
df$anti_interventionism <- rowMeans(select(df, gvslvol, gvslvue, gvcldcr, gvjbevn, gvhlthc, gvpdlwk))
describe(df$anti_interventionism)

# Social benefits bad .37 --> split
psych::alpha(select(df, sbstrec, sbprvpv, sbeqsoc, sbbsntx))
cor(select(df, sbstrec, sbprvpv, sbeqsoc, sbbsntx), use = "complete.obs")
df$benefits_eco<- rowMeans(select(df, sbstrec, sbbsntx))
df$benefits_soc<- rowMeans(select(df, sbprvpv, sbeqsoc)) 
describe(df$benefits_eco)
describe(df$benefits_soc)

# Authoritarianism .73
psych::alpha(select(df, impsafe, ipfrule, ipstrgv, ipbhprp, imptrad))
df$authoritarianism <- rowMeans(select(df, impsafe, ipfrule, ipstrgv, ipbhprp, imptrad))
describe(df$authoritarianism)

# Anti-libertarianism .68
psych::alpha(select(df, impdiff, ipadvnt, ipcrtiv, impfree, ipudrst))
df$anti_libertarianism <- rowMeans(select(df, impdiff, ipadvnt, ipcrtiv, impfree, ipudrst))
describe(df$anti_libertarianism)

# Regressive_benefits <- 0.69 / r = 0.534668
psych::alpha(select(df, regressive_taxes, earnpen, earnueb))
psych::alpha(select(df, earnpen, earnueb))
cor(select(df, earnpen, earnueb), use = "complete.obs")
df$regressive_benefits <- rowMeans(select(df, earnpen, earnueb))
describe(df$regressive_benefits)

##### Clean up and save ----
### Sort by country and id, new id variable
df <- df %>% 
  arrange(country, id) %>%
  mutate(essid = id,
         id = row_number())

### Select relevant vars
names(df)
df <- df %>%
  select(id, essid, country,  # General vars
         education, hhincome, female, age, religious, urbanization, ethnic_minority, # Demographics
         lrscale, gender_inequality, anti_lgbt, euroscepticism, anti_immigration, anti_egalitarianism, benefits_eco, benefits_soc, welfare_chauvinism, anti_interventionism, harsh_sentences, anti_mil_democracy, science_environment, government_spending, regressive_taxes, regressive_benefits, age_prejudice, authoritarianism, anti_libertarianism) # Beliefs

### Rescale beliefs to 0-1
df <- sjlabelled::remove_all_labels(df)

df <- df %>% 
  mutate(across(c(lrscale, gender_inequality, anti_lgbt, euroscepticism, anti_immigration, anti_egalitarianism, benefits_eco, benefits_soc, welfare_chauvinism, anti_interventionism, harsh_sentences, anti_mil_democracy, science_environment, government_spending, regressive_taxes, regressive_benefits, age_prejudice, authoritarianism, anti_libertarianism), 
                scales::rescale, to = c(0,1)))

### Check data 
describe(df)

### Drop everyone with more than 2 missings on the beliefs
df <- df[rowSums(is.na(df[c(11:29)])) <= 2, ]
nrow(df)

### Save to file
save(df, file = "data/df_ESS4.RData")

### Clear workspace and free up memory
rm(list = ls())
gc()