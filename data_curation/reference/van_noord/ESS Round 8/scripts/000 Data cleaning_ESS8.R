library(plyr)
library(dplyr)
library(tidyverse)
library(sjmisc)
library(psych)
library(summarytools)
library(essurvey)

##### Import ESS round 8 file ----
set_email("j.van.noord@rug.nl")
df_raw <- import_rounds(8)

temp <- df

##### Select respondents and variables, rename variables ----
df <- df_raw %>%
  filter(agea > 17 | is.na(agea)) %>% # filter out participants under 18
  select(idno, cntry, # General variables
         gndr, rlgblg, blgetmg, domicil, agea, hinctnta, eisced, # Demographics
         vote, prtvtbat:prtvtesi, # Vote variables
         lrscale, euftf, mnrgtjb, imsclbn, bnlwinc, wrkprbf, eduunmp, basinc, inctxff, sbsrnen, banhhap, ccnthum, # Beliefs
         freehms, hmsfmlsh, hmsacld, 
         imsmetn, imdfetn, impcntr, 
         gincdif, dfincac, smdfslv, 
         gvslvol, gvslvue, gvcldcr, 
         sbstrec, sbbsntx, 
         sbprvpv, sbeqsoc, 
         impsafe, ipfrule, ipbhprp, ipstrgv, imptrad, 
         impdiff, ipadvnt, ipcrtiv, impfree, ipudrst) 

df <- df %>%
  rename(id = idno, # General variables
         country = cntry, 
         female = gndr, # Demographics
         religious = rlgblg, 
         ethnic_minority = blgetmg, 
         urbanization = domicil, 
         age = agea, 
         hhincome = hinctnta, 
         education = eisced, 
         euroscepticism = euftf, # Single item beliefs (some need reverse coding)
         gender_inequality = mnrgtjb, 
         welfare_chauvinism = imsclbn, 
         anti_sb_lowincome = bnlwinc, 
         anti_sb_parents = wrkprbf, 
         spend_edu = eduunmp,
         anti_basinc = basinc, 
         anti_cli_taxes = inctxff, 
         anti_cli_renewable = sbsrnen, 
         anti_cli_ban = banhhap, 
         cli_sceptic = ccnthum) 


##### Merge country specific party vote vars into one var, and merge with popuList data ----
# Import Popu-List data
PopuList_expanded <- read.csv("data/Votebehavior_ESS_8_expanded.csv", sep = ";") 

# Merge and create vars
df <- df %>%
  mutate(party_id = coalesce(prtvtbat, prtvtcbe, prtvtfch, prtvtdcz, prtvede1, prtvtfee, prtvtdes, prtvtdfi, prtvtcfr, prtvtbgb, prtvtehu, prtvtbie, prtvtcil, prtvtbis, prtvtbit, prtvblt1, prtvtfnl, prtvtbno, prtvtdpl, prtvtcpt, prtvtdru, prtvtbse, prtvtesi)) %>%
  left_join(PopuList_expanded, by = c("country", "party_id")) %>%
  select(-c(prtvtbat:prtvtesi)) %>%
  mutate(
    vote_behavior = case_when(
      vote == 2 ~ 7,
      vote_populist == 1 & vote_farleft == 1 ~ 3,
      vote_populist == 1 & vote_farright == 1 ~ 5,
      vote_loyalty == 1 ~ 1,
      vote_farleft == 1 ~ 2,
      vote_populist == 1 ~ 4,
      vote_farright == 1 ~ 6,
      TRUE ~ NA_real_))


##### Recoding ----
### Recode 55 to NA in cli_sceptic
df$cli_sceptic[which(df$cli_sceptic == 55)] <- NA

### Reverse code
reverse_var <- function (x) (min(x, na.rm = T) + max(x, na.rm = T)) - x

df <- df %>%
  mutate(across(c(urbanization, religious, ethnic_minority, # Demographics
                  gender_inequality, anti_sb_lowincome, anti_sb_parents, cli_sceptic, euroscepticism, anti_basinc, # Single item beliefs
                  hmsfmlsh, # Anti-LGBT
                  dfincac, # Anti-egalitarianism
                  gvslvol, gvslvue, gvcldcr, # Anti-interventionism
                  sbstrec, sbbsntx, # Benefits harms economy
                  impsafe, ipfrule, ipstrgv, ipbhprp, imptrad), # Authoritarianism
                reverse_var)) # Reverse code

### Education in three subgroups
df <- df %>%
  mutate(education = case_when(
    education %in% c("1","2") ~ 1,
    education %in% c("3","4","5") ~ 2,
    education %in% c("6","7") ~ 3),
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
# Anti-LGBT .80
psych::alpha(select(df, "freehms", "hmsfmlsh", "hmsacld"))
df$anti_lgbt <- rowMeans(select(df, "freehms", "hmsfmlsh", "hmsacld"))
describe(df$anti_lgbt)

# Anti-migration .87
psych::alpha(select(df, "imsmetn", "imdfetn", "impcntr"))
df$anti_immigration <- rowMeans(select(df, "imsmetn", "imdfetn", "impcntr"))
describe(df$anti_immigration)

# Anti-Egalitarianism .59
psych::alpha(select(df, "gincdif","dfincac", "smdfslv"))
df$anti_egalitarianism <- rowMeans(select(df, "gincdif","dfincac", "smdfslv"))
describe(df$anti_egalitarianism)

# Anti-interventionism .68
psych::alpha(select(df, "gvslvol", "gvslvue","gvcldcr"))
df$anti_interventionism <- rowMeans(select(df, "gvslvol", "gvslvue","gvcldcr"))
describe(df$anti_interventionism)

# Social benefits bad .45 --> split
cor(select(df, "sbstrec", "sbprvpv", "sbeqsoc", "sbbsntx"), use = "complete.obs")
df$benefits_eco<- rowMeans(select(df, "sbstrec", "sbbsntx"))
df$benefits_soc<- rowMeans(select(df, "sbprvpv", "sbeqsoc")) 
describe(df$benefits_eco)
describe(df$benefits_soc)

# Authoritarianism .70
psych::alpha(select(df, "impsafe", "ipfrule", "ipbhprp", "ipstrgv", "imptrad"))
df$authoritarianism <- rowMeans(select(df, "impsafe", "ipfrule", "ipbhprp", "ipstrgv", "imptrad"))
describe(df$authoritarianism)

# Anti-libertarianism .66
psych::alpha(select(df, "impdiff", "ipadvnt", "ipcrtiv", "impfree", "ipudrst"))
df$anti_libertarianism <- rowMeans(select(df, "impdiff", "ipadvnt", "ipcrtiv", "impfree", "ipudrst"))
describe(df$anti_libertarianism)


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
         vote_behavior, # Vote var
         lrscale, gender_inequality, anti_lgbt, euroscepticism, anti_immigration, anti_egalitarianism, benefits_eco, benefits_soc, welfare_chauvinism, anti_interventionism, anti_sb_lowincome, anti_sb_parents, spend_edu, anti_basinc, anti_cli_taxes, anti_cli_renewable, anti_cli_ban, cli_sceptic, authoritarianism, anti_libertarianism) # Beliefs

### Rescale beliefs to 0-1
df <- sjlabelled::remove_all_labels(df)

df <- df %>% 
  mutate(across(c(lrscale, gender_inequality, anti_lgbt, euroscepticism, anti_immigration, anti_egalitarianism, benefits_eco, benefits_soc, welfare_chauvinism, anti_interventionism, anti_sb_lowincome, anti_sb_parents, spend_edu, anti_basinc, anti_cli_taxes, anti_cli_renewable, anti_cli_ban, cli_sceptic, authoritarianism, anti_libertarianism), 
                scales::rescale, to = c(0,1)))

### Check data 
describe(df)

### Drop everyone with more than 2 missings on the beliefs
df <- df[rowSums(is.na(df[c(12:31)])) <= 2, ]


### Save to file
save(df, file = "data/df_ESS8.RData")

### Clear workspace and free up memory
rm(list = ls())
gc()