library(tidyverse)
library(dplyr)
library(ggplot2)

data <- read.csv("Access_to_a_Livable_Planet_Dataset_Clean.csv")

dataConverted <- data %>% 
  mutate(
Days.PM2.5 = as.numeric(Days.PM2.5), 
Days.PM10 = as.numeric(Days.PM10), 
Days.CO = as.numeric(Days.CO), 
Days.NO2 = as.numeric(Days.NO2), 
Days.Ozone = as.numeric(Days.Ozone) )

model <- lm( Max.AQI ~ Days.PM2.5 + Days.PM10 + Days.CO + Days.NO2 + Days.Ozone, 
               data = df ) 

pvals <- summary(model_aqi)$coefficients[, 4] 
pvals
