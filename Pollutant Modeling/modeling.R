#Modeling pollutant

library(tidyverse)
library(dplyr)
library(ggplot2)

data <- read.csv("Access_to_a_Livable_Planet_Dataset_Clean.csv")

getwd()
setwd("C:/Users/rosar/Datathon/Data")

# avgByYear <- function(data, year, pollutant_col) { 
#     data %>% 
#         #drop_na() %>%
#         mutate(
#             {{ pollutant_col }} := as.numeric({{ pollutant_col }})
#         ) %>%
#         group_by({{ year }}) %>% 
#         summarize(avg = mean({{ pollutant_col }}, na.rm = TRUE)) }

# avgByYear(data, Year, `Days.CO`)

names(data)
unique(data$State)

# calculate probability of each pollutant's risk using aqi
# number of states with an aqi over 100 / total number of states



