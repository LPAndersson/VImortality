# This file calculates the log-scores for some other models using the StMoMo
# library.

library(StMoMo)
library(tidyverse)
library(demography)


source("log_scores.R")

country <- 'sweden'
sex <- 'Male'
latent_dim <- 0
max_age <- 100

df <- tibble(
  country = character(),
  first_year_train = numeric(),
  last_year_train = numeric(),
  sex = character() ,
  max_age = numeric(),
  forecast_horizon = numeric(),
  log_score = numeric(),
  date = character(),
  time = character(),
  model = character()
)

load("SWdata.dat")

SWStMoMo <- StMoMoData(SWdata, series = "male")
SWStMoMo$Dxt <- floor(SWStMoMo$Dxt)

#Calculate average log-score

first_year <- 1931
train_length <- 60
forecast_length <- 10
last_year <- 2021

start_years = first_year:( last_year - train_length - forecast_length + 1)

log_score_fcn <- log_score_LC
model <- 'LC'

for (start_year_idx in seq_along(start_years)) {
  print(start_years[start_year_idx])
  log_scores <- log_score_fcn(
    SWStMoMo,
    start_years[start_year_idx],
    start_years[start_year_idx] + train_length - 1,
    10    )

  for(fcast_horizon in 1:length(log_scores)){
    df <- df %>% add_row(
      country = country,
      first_year_train = start_years[start_year_idx],
      last_year_train = start_years[start_year_idx] + train_length - 1,
      sex = sex ,
      max_age = max_age,
      forecast_horizon = fcast_horizon,
      log_score = log_scores[fcast_horizon],
      date = format(Sys.time(), "%d/%M/%Y"),
      time = format(Sys.time(), "%X"),
      model = model
    )
  }

  write_csv(df, 'logScoreMomo.csv' )

}

log_score_fcn <- log_score_M7
model <- 'M7'
for (start_year_idx in seq_along(start_years)) {
  print(start_years[start_year_idx])
  log_scores <- log_score_fcn(
    SWStMoMo,
    start_years[start_year_idx],
    start_years[start_year_idx] + train_length - 1,
    10    )

  for(fcast_horizon in 1:length(log_scores)){
    df <- df %>% add_row(
      country = country,
      first_year_train = start_years[start_year_idx],
      last_year_train = start_years[start_year_idx] + train_length - 1,
      sex = sex ,
      max_age = max_age,
      forecast_horizon = fcast_horizon,
      log_score = log_scores[fcast_horizon],
      date = format(Sys.time(), "%d/%M/%Y"),
      time = format(Sys.time(), "%X"),
      model = model
    )
  }

  write_csv(df, 'logScoreMomo.csv' )

}

log_score_fcn <- log_score_plat
model <- 'PLAT'
for (start_year_idx in seq_along(start_years)) {
  print(start_years[start_year_idx])
  log_scores <- log_score_fcn(
    SWStMoMo, 
    start_years[start_year_idx], 
    start_years[start_year_idx] + train_length - 1, 
    10    )
  
  for(fcast_horizon in 1:length(log_scores)){
    df <- df %>% add_row(
      country = country,
      first_year_train = start_years[start_year_idx],
      last_year_train = start_years[start_year_idx] + train_length - 1,
      sex = sex ,
      max_age = max_age,
      forecast_horizon = fcast_horizon,
      log_score = log_scores[fcast_horizon],
      date = format(Sys.time(), "%d/%M/%Y"),
      time = format(Sys.time(), "%X"),
      model = model
    )
  }
  
  write_csv(df, 'logScoreMomo.csv' )
  
}

write_csv(df, 'logScoreMomo.csv' )








