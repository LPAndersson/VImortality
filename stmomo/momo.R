library(StMoMo)
library(tidyverse)
library(demography)

source("log_scores.R")

country <- 'sweden'
sex <- 'Male'
nn_layers <- 0
latent_dim <- 0
max_age <- 100

df <- tibble(
  Country = character(),
  first_year_train = numeric(),
  last_year_train = numeric(),
  Sex = character() ,
  max_age = numeric(),
  nn_layers = numeric(),
  latent_dim = numeric(),
  'Forecast horizon' = numeric(),
  'Log score' = numeric(),
  Date = character(),
  Time = character(),
  model = character()
)

load("SWdata.dat")

SWStMoMo <- StMoMoData(SWdata, series = "male")
SWStMoMo$Dxt <- floor(SWStMoMo$Dxt)

#Calculate average log-score

first_year <- 1920
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
      Country = country,
      first_year_train = start_years[start_year_idx],
      last_year_train = start_years[start_year_idx] + train_length - 1,
      Sex = sex ,
      max_age = max_age,
      nn_layers = nn_layers,
      latent_dim = latent_dim,
      'Forecast horizon' = fcast_horizon,
      'Log score' = log_scores[fcast_horizon],
      Date = format(Sys.time(), "%d/%M/%Y"),
      Time = format(Sys.time(), "%X"),
      model = model
    )
  }
  
  write_csv(df, 'logScoreOtherModels.csv' )
  
}

log_score_fcn <- log_score_plat
model <- 'Plat'

for (start_year_idx in seq_along(start_years)) {
  print(start_years[start_year_idx])
  log_scores <- log_score_fcn(
    SWStMoMo, 
    start_years[start_year_idx], 
    start_years[start_year_idx] + train_length - 1, 
    10    )
  
  for(fcast_horizon in 1:length(log_scores)){
    df <- df %>% add_row(
      Country = country,
      first_year_train = start_years[start_year_idx],
      last_year_train = start_years[start_year_idx] + train_length - 1,
      Sex = sex ,
      max_age = max_age,
      nn_layers = nn_layers,
      latent_dim = latent_dim,
      'Forecast horizon' = fcast_horizon,
      'Log score' = log_scores[fcast_horizon],
      Date = format(Sys.time(), "%d/%M/%Y"),
      Time = format(Sys.time(), "%X"),
      model = model
    )
  }
  
  write_csv(df, 'logScoreOtherModels.csv' )
  
}

write_csv(df, 'logScoreOtherModels.csv' )








