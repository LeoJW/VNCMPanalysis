library(tidyverse)
library(BayesianFirstAid)

data_dir <- "C:/Users/lwood39/Documents/ResearchPhD/VNCMP/data_for_python/"

df = read.csv(paste(data_dir, "velocity_estimates_all_units.csv", sep="")) %>% 
  as_tibble() %>% 
  mutate(vel = dx / dt)

extract_model_results <- function(df, y){
  mod <- bayes.binom.test(c(df$n_descend, df$n_total))
  result <- as.data.frame(t(mod[["stats"]][1,]))
}

newdf <- df %>%
  group_by(moth, unit, globalunit) %>% 
  summarize(
    n_descend = sum(vel < 0),
    n_ascend = sum(vel > 0),
    n_total = n()
  ) %>% 
  group_by(moth, unit, globalunit) %>% 
  group_modify(extract_model_results) %>% 
  ungroup() %>% 
  write_csv(paste(data_dir, "direction_estimate_stats_all_units.csv", sep=""))


