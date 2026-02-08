#Modeling pollutant

library(tidyverse)
library(dplyr)
library(ggplot2)

data <- read.csv("Access_to_a_Livable_Planet_Dataset_Clean.csv")

getwd()
setwd("C:/Users/rosar/Datathon/Data")

names(data)
unique(data$State)

#calculate percentage of days that each pollutant was a main factor divided by 365 days for each county
percentageDays <- function(data, county_col, pollutant_col) {
  data %>%
    group_by({{ county_col }}) %>%
    summarize(
      percentage = sum(as.numeric({{ pollutant_col }}), na.rm = TRUE) / 365 * 100,
      .groups = "drop"
    )
}

print(percentageDays(data, `County`, `Days.PM10`))

# calculate which pollutant poses the high longest risk by adding the percentages of each county and taking the average
pollutants <- c("Days.CO", "Days.NO2", "Days.Ozone", "Days.PM2.5", "Days.PM10")
average_percentage <- data.frame()

for (pollutant in pollutants) {
  percentage <- percentageDays(data, County, !!sym(pollutant))
  average_percentage <- rbind(
    average_percentage,
    data.frame(
      Pollutant = pollutant,
      Average_Percentage = mean(percentage$percentage, na.rm = TRUE)
    )
  )
}

# Print once
print(average_percentage)




