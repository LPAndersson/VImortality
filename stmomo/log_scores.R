log_score_plat <- function(data, train_from, train_to, forecast_length){
  
  ages.fit <- 0:100
  
  test_years <- (train_to+1):(train_to + forecast_length)
  
  nsim <- 10000
  
  f2 <- function(x, ages) mean(ages) - x
  
  f3 <- function(x, ages) pmax(mean(ages)-x,0)
  
  constPlat <- function(ax, bx, kt, b0x, gc, wxt, ages){
    nYears <- dim(wxt)[2]
    x <- ages
    t <- 1:nYears
    c <- (1 - tail(ages, 1)):(nYears - ages[1])
    xbar <- mean(x)
    #\sum g(c)=0, \sum cg(c)=0, \sum c^2g(c)=0
    phiReg <- lm(gc ~ 1 + c + I(c^2), na.action = na.omit)
    phi <- coef(phiReg)
    gc <- gc - phi[1] - phi[2] * c - phi[3] * c^2
    kt[2, ] <- kt[2, ] + 2 * phi[3] * t
    kt[1, ] <- kt[1, ] + phi[2] * t + phi[3] * (t^2 - 2 * xbar * t)
    ax <- ax + phi[1] - phi[2] * x + phi[3] * x^2
    #\sum kt[i, ] = 0
    ci <- rowMeans(kt, na.rm = TRUE)
    ax <- ax + ci[1] + ci[2] * (xbar - x) + ci[3] * pmax(xbar - x, 0)
    kt[1, ] <- kt[1, ] - ci[1]
    kt[2, ] <- kt[2, ] - ci[2]
    kt[3, ] <- kt[3, ] - ci[3]
    list(ax = ax, bx = bx, kt = kt, b0x = b0x, gc = gc)
  }
  
  PLAT <- StMoMo(link = "log", staticAgeFun = TRUE,
                 periodAgeFun = c("1", f2, f3), cohortAgeFun = "1",
                 constFun = constPlat)
  
  ages.fit <- 0:100
  
  wxt <- genWeightMat(ages = ages.fit, years = train_from:train_to, clip = 3)
  
  
  PLATfit <- fit(
    PLAT, 
    data = data, 
    ages.fit = ages.fit, 
    years.fit = train_from:train_to,
    wxt = wxt
  )
  
  PLATsim <- simulate(
    PLATfit, 
    nsim = nsim, 
    h = forecast_length, 
    gc.order = c(2, 0, 0)
  )
  
  log_score <- array(data = 0.0, dim = c(nsim, length(ages.fit)))
  log_score_t <- array(data = 0.0, dim = forecast_length)
  
  for (t in seq_len(forecast_length)) {
    for (i in seq_len(nsim)) {
      for (age in seq_len(length(ages.fit))) {
        year <- PLATsim$years[t]
        year_idx <- which(data$years == year)
        intensity <- PLATsim$rates[age,t,i]
        exposure <- data$Ext[age,year_idx]
        deaths <- data$Dxt[age,year_idx]
        log_score[i,age] <- dpois(deaths, intensity * exposure, log = TRUE)
      }
    }
    
    log_score_t[t] <- sum( matrixStats::colLogSumExps(log_score) - log(nsim) )
    
  }
  
  log_score_t
}


log_score_LC <- function(data, train_from, train_to, forecast_length){
  
  ages.fit <- 0:100
  
  test_years <- (train_to+1):(train_to + forecast_length)
  
  nsim <- 10000
  
  mortModel <- lc(link = "log")
  
  wxt <- genWeightMat(ages = ages.fit, years = train_from:train_to, clip = 3)
  
  
  mortFit <- fit(
    mortModel, 
    data = data, 
    ages.fit = ages.fit, 
    years.fit = train_from:train_to,
    wxt = wxt
  )
  
  mortSim <- simulate(
    mortFit, 
    nsim = nsim, 
    h = forecast_length, 
    gc.order = c(2, 0, 0)
  )
  
  log_score <- array(data = 0.0, dim = c(nsim, length(ages.fit)))
  log_score_t <- array(data = 0.0, dim = forecast_length)
  
  for (t in seq_len(forecast_length)) {
    for (i in seq_len(nsim)) {
      for (age in seq_len(length(ages.fit))) {
        year <- mortSim$years[t]
        year_idx <- which(data$years == year)
        intensity <- mortSim$rates[age,t,i]
        exposure <- data$Ext[age,year_idx]
        deaths <- data$Dxt[age,year_idx]
        log_score[i,age] <- dpois(deaths, intensity * exposure, log = TRUE)
      }
    }
    
    log_score_t[t] <- sum( matrixStats::colLogSumExps(log_score) - log(nsim) )
    
  }
  
  log_score_t
}


log_score_M7 <- function(data, train_from, train_to, forecast_length){
  
  ages.fit <- 0:100
  
  test_years <- (train_to+1):(train_to + forecast_length)
  
  nsim <- 10000
  
  mortModel <- m7(link = "log")
  
  wxt <- genWeightMat(ages = ages.fit, years = train_from:train_to, clip = 3)
  
  mortFit <- fit(
    mortModel, 
    data = data, 
    ages.fit = ages.fit, 
    years.fit = train_from:train_to,
    wxt = wxt
  )
  
  mortSim <- simulate(
    mortFit, 
    nsim = nsim, 
    h = forecast_length, 
    gc.order = c(2, 0, 0)
  )
  
  log_score <- array(data = 0.0, dim = c(nsim, length(ages.fit)))
  log_score_t <- array(data = 0.0, dim = forecast_length)
  
  for (t in seq_len(forecast_length)) {
    for (i in seq_len(nsim)) {
      for (age in seq_len(length(ages.fit))) {
        year <- mortSim$years[t]
        year_idx <- which(data$years == year)
        intensity <- mortSim$rates[age,t,i]
        exposure <- data$Ext[age,year_idx]
        deaths <- data$Dxt[age,year_idx]
        log_score[i,age] <- dpois(deaths, intensity * exposure, log = TRUE)
      }
    }
    
    log_score_t[t] <- sum( matrixStats::colLogSumExps(log_score) - log(nsim) )
    
  }
  
  log_score_t
}

log_score_cbd <- function(data, train_from, train_to, forecast_length){
  
  ages.fit <- 0:100
  
  test_years <- (train_to+1):(train_to + forecast_length)
  
  nsim <- 10000
  
  mortModel <- cbd(link = "log")
  
  wxt <- genWeightMat(ages = ages.fit, years = train_from:train_to, clip = 3)
  
  mortFit <- fit(
    mortModel, 
    data = data, 
    ages.fit = ages.fit, 
    years.fit = train_from:train_to,
    wxt = wxt
  )
  
  mortSim <- simulate(
    mortFit, 
    nsim = nsim, 
    h = forecast_length, 
    gc.order = c(2, 0, 0)
  )
  
  log_score <- array(data = 0.0, dim = c(nsim, length(ages.fit)))
  log_score_t <- array(data = 0.0, dim = forecast_length)
  
  for (t in seq_len(forecast_length)) {
    for (i in seq_len(nsim)) {
      for (age in seq_len(length(ages.fit))) {
        year <- mortSim$years[t]
        year_idx <- which(data$years == year)
        intensity <- mortSim$rates[age,t,i]
        exposure <- data$Ext[age,year_idx]
        deaths <- data$Dxt[age,year_idx]
        log_score[i,age] <- dpois(deaths, intensity * exposure, log = TRUE)
      }
    }
    
    log_score_t[t] <- sum( matrixStats::colLogSumExps(log_score) - log(nsim) )
    
  }
  
  log_score_t
}


