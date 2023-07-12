set.seed(42)
library(lme4)
library(mgcv)
library(itsadug)
library(ggplot2)
library(dplyr)
library(gamlss)
library(betareg)
library(sjPlot)
library(DescTools)

prep_df_analysis <- function(df){
  df$function. = factor(df$function., levels = c('entropy', 'rho', 'csd'))
  df$status = factor(df$status, levels = c('noise', 'sync20', 'sync50', 'sync70', 'sync100')) 
  df$window = factor(df$window) 
  df$coup_per = factor(df$coup_per, levels = c('False', 'True'))
  df$sync_laps = as.numeric(df$sync_laps)
  df['distance.norm'] <- ave(df$distance, df$function., FUN=function(x) x/max(x))
  df$distance.norm <- replace(df$distance.norm, df$distance.norm==0, 0.0001)
  df$distance.norm <- replace(df$distance.norm, df$distance.norm==1, 0.9999)
  #df<-df[!(df$status=="noise"),]
  return(df)
}

# CLEAN signal analysis ####
## Logistic regression for coupled signal (long data)####
df <- read.csv("data/cleanSignal_df.csv")
df_analysis <- prep_df_analysis(df)

glm.COUP_base <- glm(coup_per ~1,
                family=binomial(link='logit'),
                data=df_analysis)
glm.COUP_status <- glm(coup_per ~status,
                  family=binomial(link='logit'),
                  data=df_analysis)
anova(glm.COUP_base, glm.COUP_status, test='Chisq')
glm.COUP_status.window <- glm(coup_per ~ status + window, 
                         family=binomial(link='logit'),
                         data=df_analysis)
anova(glm.COUP_status, glm.COUP_status.window, test='Chisq')
glm.COUP_status.window.function <- glm(coup_per ~ status + window + function., 
                              family=binomial(link='logit'),
                              data=df_analysis)
anova(glm.COUP_status.window, glm.COUP_status.window.function, test='Chisq')

summary(glm.COUP_status.window.function)
with(summary(glm.COUP_status.window.function), 1 - deviance/null.deviance)
PseudoR2(glm.COUP_status.window.function, 'McFadden')
tab_model(glm.COUP_status.window.function, p.style= 'stars')


##  beta regression for normalised distance ####
df_analysis_b <- df_analysis[df_analysis$status != 'noise',]
df_analysis_b$status = factor(df_analysis_b$status, levels = c('sync20', 'sync50', 'sync70', 'sync100')) 

beta.DIST_base <- gamlss(distance.norm ~ status, 
                         family = BE,
                         data = df_analysis_b)

beta.DIST_window <- gamlss(distance.norm ~ status + window, 
                           family = BE,
                           data = df_analysis_b)
LR.test(beta.DIST_base, beta.DIST_window)
beta.DIST_window.function <- gamlss(distance.norm ~ status + window + function., 
                                    family = BE,
                                    data = df_analysis_b)
LR.test(beta.DIST_window, beta.DIST_window.function, print = TRUE)
summary(beta.DIST_window.function)
Rsq(beta.DIST_window.function)
round(exp(cbind(OR = coef(beta.DIST_window.function), confint(beta.DIST_window.function))),2)

## Check distance ability to distinguish noise from sync ####
df_analysis_c <- df_analysis
sync <- c()
for (v in df_analysis$status){
  if (v == 'noise'){
    sync <- append(sync, 0)
  } else {
    sync <- append(sync, 1)
    }
}
df_analysis_c$sync <- sync

glm.SYNC_distance.window.function <- glm(sync ~ distance.norm + window + function., 
                                       family=binomial(link='logit'),
                                       data=df_analysis_c)

summary(glm.SYNC_distance.window.function)
PseudoR2(glm.SYNC_distance.window.function, 'McFadden')
tab_model(glm.SYNC_distance.window.function, p.style= 'stars')


# NOISED signal analysis ####
df_noised <- read.csv("data/noisedSignal_df.csv")
df_analysis_noised <- prep_df_analysis(df_noised)

## Logistic regression for coupled signal (long data)####
glm.COUP_base_noised <- glm(coup_per ~1,
                     family=binomial(link='logit'),
                     data=df_analysis_noised)
glm.COUP_status_noised <- glm(coup_per ~status,
                       family=binomial(link='logit'),
                       data=df_analysis_noised)
anova(glm.COUP_base_noised, glm.COUP_status_noised, test='Chisq')
glm.COUP_status.window_noised <- glm(coup_per ~ status + window, 
                              family=binomial(link='logit'),
                              data=df_analysis_noised)
anova(glm.COUP_status_noised, glm.COUP_status.window_noised, test='Chisq')
glm.COUP_status.window.function_noised <- glm(coup_per ~ status + window + function., 
                                                family=binomial(link='logit'), 
                                              data=df_analysis_noised)
anova(glm.COUP_status.window_noised, glm.COUP_status.window.function_noised, test='Chisq')
summary(glm.COUP_status.window.function_noised)
PseudoR2(glm.COUP_status.window_noised, 'McFadden')
round(exp(cbind(OR = coef(glm.COUP_status.window.function_noised), confint(glm.COUP_status.window.function_noised))),2)
tab_model(glm.COUP_status.window.function_noised, p.style= 'stars')


## Beta regression for normalised distance ####
df_analysis_noised_b <- df_analysis_noised[df_analysis_noised$status != 'noise',]
df_analysis_noised_b$status = factor(df_analysis_noised_b$status, levels = c('sync20', 'sync50', 'sync70', 'sync100')) 

beta.DIST_base_noised <- gamlss(distance.norm ~ status, 
                                family = BE,
                                data = df_analysis_noised_b)

beta.DIST_window_noised <- gamlss(distance.norm ~ status + window, 
                                  family = BE,
                                  data = df_analysis_noised_b)
LR.test(beta.DIST_base_noised, beta.DIST_window_noised, print = TRUE)
beta.DIST_window.function_noised <- gamlss(distance.norm ~ status + window + function., 
                                           family = BE,
                                           data = df_analysis_noised_b)
LR.test(beta.DIST_window_noised, beta.DIST_window.function_noised, print = TRUE)
summary(beta.DIST_window.function_noised)
Rsq(beta.DIST_window.function_noised)
round(exp(cbind(OR = coef(beta.DIST_window.function_noised), confint(beta.DIST_window.function_noised))),2)


## Check distance ability to distinguish noise from sync ####
df_analysis_noised_c <- df_analysis_noised
sync <- c()
for (v in df_analysis_noised$status){
  if (v == 'noise'){
    sync <- append(sync, 0)
  } else {
    sync <- append(sync, 1)
  }
}
df_analysis_noised_c$sync <- sync

glm.SYNC_distance.window.function_noised <- glm(sync ~ distance.norm + window + function., 
                                         family=binomial(link='logit'),
                                         data=df_analysis_noised_c)

summary(glm.SYNC_distance.window.function_noised)
PseudoR2(glm.SYNC_distance.window.function_noised, 'McFadden')
tab_model(glm.SYNC_distance.window.function_noised, p.style= 'stars')

###############










