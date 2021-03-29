import argparse
import time
from os.path import exists

import copy

import numpy as np
import torch
import torch.nn as nn

from scipy.stats import poisson
from scipy.special import logsumexp

import scipy

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceEnum_ELBO, TraceTMC_ELBO, config_enumerate
from pyro.optim import ClippedAdam
from pyro.optim import RMSprop

from mortalityForecast import utils

class Emitter(nn.Module):
    #Parameterizes `p(x_t | z_t)`

    def __init__(self, input_dim, latent_dim = 3, nn_dim = 32, nn_layers = 2):
        super(Emitter, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.nn_dim = nn_dim
        self.nn_layers = nn_layers

        if nn_layers == 3:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, nn_dim),
                nn.ReLU(),
                nn.Linear(nn_dim, nn_dim),
                nn.ReLU(),
                nn.Linear(nn_dim, input_dim))
        elif nn_layers == 2:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, nn_dim),
                nn.ReLU(),
                nn.Linear(nn_dim, input_dim))
        elif nn_layers == 1:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, input_dim))


    def forward(self, x):
        return torch.exp(self.decoder(x))

class LevelTrend(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    """

    def __init__(self, latent_dim = 3):
        super(LevelTrend, self).__init__()

        self.latent_dim = latent_dim
        self.dampening = nn.Parameter(torch.zeros(latent_dim))
        self.trend_mean = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, level_prev, trend_prev):

        damp = utils.sigmoid(self.dampening, min = 0.0, max = 1.0)
        trend_mean = self.trend_mean

        return (level_prev + trend_prev, trend_mean + damp * (trend_prev - trend_mean ) )

    def forward_cov(self, level_scale, trend_scale, covariance, level_noise_scale, trend_noise_scale):

        damp = utils.sigmoid(self.dampening, min = 0.0, max = 1.0)

        level_scale_next = torch.sqrt( level_scale**2 + trend_scale**2 + 2*covariance + level_noise_scale**2 )
        trend_scale_next = torch.sqrt( (damp * trend_scale)**2 + trend_noise_scale**2 )
        covariance_next = damp * covariance + damp * ( trend_scale**2 )

        return (level_scale_next, trend_scale_next, covariance_next)



class _DMM(nn.Module):

    def __init__(self, num_years, input_dim, emitter, latent_transition, use_cuda = False):
        super(_DMM, self).__init__()
        
        self.num_years = num_years
        self.input_dim = input_dim
        self.use_cuda = use_cuda

        self.emitter = emitter
        self.latent_transition = latent_transition

        self.latent_dim = self.emitter.latent_dim

        self.scale_transform = nn.Softplus()
        self.scale_transform_inv = utils.softplus_inv

        #Model parameters
        self.level_0 = nn.Parameter(torch.zeros(self.latent_dim))
        self.trend_0 = nn.Parameter(torch.zeros(self.latent_dim))
        self.level_scale = nn.Parameter(torch.ones(self.latent_dim))
        self.trend_scale = nn.Parameter(torch.ones(self.latent_dim))

        #Guide parameters
        self.guide_level_mean = nn.Parameter(torch.zeros(self.latent_dim,num_years))
        self.guide_level_scale = nn.Parameter(torch.ones(self.latent_dim,num_years))
        self.guide_level_damp = nn.Parameter(torch.zeros(self.latent_dim,num_years-1))
        self.guide_trend_mean = nn.Parameter(torch.zeros(self.latent_dim,num_years))
        self.guide_trend_scale = nn.Parameter(torch.ones(self.latent_dim,num_years))
        self.guide_trend_damp = nn.Parameter(torch.zeros(self.latent_dim,num_years-1))
        self.guide_corr = nn.Parameter(torch.zeros(self.latent_dim,num_years))

        if use_cuda:
            self.cuda()

    def model(self, data):

        deaths, exposure = data
        
        pyro.module("dmm", self)

        level_prev = self.level_0
        trend_prev = self.trend_0

        for t in range(self.num_years):

            (level_loc, trend_loc) = self.latent_transition(level_prev, trend_prev)
                
            # scale is standard deviation    
            level = pyro.sample("level_%d" % t, dist.Normal(level_loc, self.scale_transform(self.level_scale) ).to_event(1))
            trend = pyro.sample("trend_%d" % t, dist.Normal(trend_loc, self.scale_transform(self.trend_scale) ).to_event(1))

            # compute the intensities that parameterize the Poisson likelihood
            intensity_t = self.emitter(level)
            
            for i in range( len(deaths[t, :]) ):
                if exposure[t, i] > 0 and intensity_t[i] > 0 :
                    pyro.sample("obs_%d_%d" % (t, i+1),
                            dist.Poisson(intensity_t[i] * exposure[t, i], validate_args = False),
                            obs=deaths[t, i])

            level_prev = level
            trend_prev = trend

    def guide(self, data):

        deaths, exposure = data

        level_vec =  torch.zeros(self.latent_dim, self.num_years)
        trend_vec = torch.zeros(self.latent_dim, self.num_years)

        pyro.module("dmm", self)

        level_mean = self.guide_level_mean[:,0]
        level_scale = self.scale_transform(self.guide_level_scale[:,0])

        level = pyro.sample("level_0", 
                            dist.Normal(level_mean, level_scale)
                            .to_event(1))

        corr = utils.sigmoid(self.guide_corr[:,0], min = -1.0, max = 1.0)

        trend_mean = self.guide_trend_mean[:,0] + corr * level
        trend_scale = self.scale_transform(self.guide_trend_scale[:,0])

        trend = pyro.sample("trend_0", 
                            dist.Normal(trend_mean, trend_scale )
                            .to_event(1))

        level_vec[:,0] = level
        trend_vec[:,0] = trend

        for t in range(1, self.num_years):

            corr = utils.sigmoid(self.guide_corr[:,t], min = -1.0, max = 1.0)
            
            level_damp = utils.sigmoid(self.guide_level_damp[:,t-1], min = 0.0, max = 1.0)
            trend_damp = utils.sigmoid(self.guide_trend_damp[:,t-1], min = 0.0, max = 1.0)

            level_mean = self.guide_level_mean[:,t] + level_damp * level
            level_scale = self.scale_transform(self.guide_level_scale[:,t])

            level = pyro.sample("level_%d" % (t), 
                              dist.Normal(level_mean, level_scale )
                              .to_event(1))

            trend_mean = self.guide_trend_mean[:,t] + trend_damp * trend + corr * level
            trend_scale = self.scale_transform(self.guide_trend_scale[:,t])

            trend = pyro.sample("trend_%d" % (t), 
                              dist.Normal(trend_mean, trend_scale )
                              .to_event(1))

            level_vec[:,t] = level
            trend_vec[:,t] = trend

        return (level_vec, trend_vec)

class Mortality:

    def __init__(self, param, cuda = False):

        self.emitter = Emitter(input_dim = param['max_age'] + 1, latent_dim = param['latent_dim'], nn_layers = param['nn_layers'])
        
        self.latent_transition = LevelTrend(latent_dim = param['latent_dim'])

        self.latent_dim = param['latent_dim']
        self.num_train_years = param['last_year_train'] - param['first_year_train'] + 1
        self.num_test_years = param['last_year_test'] - param['first_year_test'] + 1

        self.cuda = cuda

    def fit(self, exposure, deaths, num_steps = 1000, log_freq = 1, checkpoint_freq = 10, load_file = None, save_file = None):

        self.num_ages = exposure.shape[1]
        self.num_years = exposure.shape[0]

        exposure = torch.from_numpy(exposure)
        deaths = torch.from_numpy(deaths)
            
        if self.cuda:
            exposure = exposure.cuda()
            deaths = deaths.cuda()

        self.data = (deaths, exposure)

        pyro.clear_param_store()

        self.dmm = _DMM(num_years = self.num_years, input_dim = self.num_ages, emitter = self.emitter, latent_transition  = self.latent_transition, use_cuda = self.cuda)

        #self.optimizer = RMSprop({"lr": 0.01})
        #self.optimizer = Adagrad({"lr": 0.1})

        adam_params = {"lr": 0.0003, "betas": (0.96, 0.999),
               "clip_norm": 10.0, "lrd": 0.99996,
               "weight_decay": 2.0}
        optimizer = ClippedAdam(adam_params)

        if load_file is not None:
            self.load_checkpoint(load_file)

        svi = SVI(self.dmm.model, self.dmm.guide, self.optimizer, loss=Trace_ELBO())

        vi_times = [time.time()]

        self.train_loss = []
        loss = 0.0

        for step in range(num_steps):

            loss += svi.step(self.data)

            if (step + 1) % log_freq == 0: # report training diagnostics
                vi_times.append(time.time())
                self.train_loss.append(loss/checkpoint_freq)
                epoch_time = vi_times[-1] - vi_times[-2]
                
                print("[step {:04}]  {:.4e} \t(dt = {:.4f} sec \t t =  {:.1f}) min".format(step+1, loss/log_freq, epoch_time, (vi_times[-1] - vi_times[0])/60))
                
                loss = 0.0

            if (step + 1) % log_freq == 0 and save_file is not None:
                self.save_checkpoint(save_file)

        return self

    def load(self, file_name, param, exposure, deaths):

        self.param = param

        self.num_ages = exposure.shape[1]
        self.num_years = exposure.shape[0]

        exposure = torch.from_numpy(exposure)
        deaths = torch.from_numpy(deaths)
            
        if self.cuda:
            exposure = exposure.cuda()
            deaths = deaths.cuda()

        self.data = (deaths, exposure)

        self.dmm = _DMM(num_years = self.num_years, input_dim = self.num_ages, emitter = self.emitter, latent_transition  = self.latent_transition)

        self.optimizer = RMSprop({"lr": 0.01})

        self.load_checkpoint(file_name)

        return self

    def load_checkpoint(self, file):

        load_model = file + ".model"
        load_opt = file + ".opt"

        assert exists(load_opt) and exists(load_model), \
            "--load-model and/or --load-opt misspecified"
        self.dmm.load_state_dict(torch.load(load_model))
        self.optimizer.load(load_opt)

    def save_checkpoint(self, file):

        save_model = file + ".model"
        save_opt = file + ".opt"

        torch.save(self.dmm.state_dict(), save_model)
        self.optimizer.save(save_opt)

    @property
    def level_loc(self):

        level_mean = self.dmm.guide_level_mean

        out = torch.zeros((self.dmm.latent_dim, self.dmm.num_years))
        out[:,0] = level_mean[:,0]

        for i in range(self.dmm.num_years-1):
            level_damp = utils.sigmoid(self.dmm.guide_level_damp[:,i], min = 0.0, max = 1.0)
            out[:,i+1] = level_mean[:,i+1] + level_damp * out[:,i]

        return out.detach()

    @property
    def level_scale(self):

        level_scale = self.dmm.scale_transform(self.dmm.guide_level_scale)

        out = torch.zeros((self.dmm.latent_dim, self.dmm.num_years))
        out[:,0] = level_scale[:,0]

        for i in range(self.dmm.num_years-1):
            level_damp = utils.sigmoid(self.dmm.guide_level_damp[:,i], min = 0.0, max = 1.0)
            out[:,i+1] = torch.sqrt(level_scale[:,i+1]**2 + (level_damp * out[:,i])**2)

        return out.detach()

    @property
    def level_trend_cov(self):
        
        corr = utils.sigmoid(self.dmm.guide_corr[:,0], min = -1.0, max = 1.0)
        level_scale = self.dmm.scale_transform(self.dmm.guide_level_scale[:,0])

        out = torch.zeros((self.dmm.latent_dim, self.dmm.num_years))
        out[:,0] = corr * (level_scale**2)

        for i in range(self.dmm.num_years-1):
            corr = utils.sigmoid(self.dmm.guide_corr[:,i+1], min = -1.0, max = 1.0)

            trend_damp = utils.sigmoid(self.dmm.guide_trend_damp[:,i], min = 0.0, max = 1.0)
            level_damp = utils.sigmoid(self.dmm.guide_level_damp[:,i], min = 0.0, max = 1.0)
            
            level_scale = self.dmm.scale_transform(self.dmm.guide_level_scale[:,i+1])

            out[:,i+1] = level_damp*trend_damp*out[:,i] + corr * level_damp * level_scale**2

        return out.detach()

    @property
    def trend_loc(self):

        trend_mean = self.dmm.guide_trend_mean 
        corr = utils.sigmoid(self.dmm.guide_corr[:,0], min = -1.0, max = 1.0)

        level_mean = self.level_loc

        out = torch.zeros((self.dmm.latent_dim,self.dmm.num_years))
        out[:,0] = trend_mean[:,0] + corr * level_mean[:,0]

        for i in range(self.dmm.num_years-1):
            corr = utils.sigmoid(self.dmm.guide_corr[:,i+1], min = -1.0, max = 1.0)
            trend_damp = utils.sigmoid(self.dmm.guide_trend_damp[:,i], min = 0.0, max = 1.0)
            out[:,i+1] = trend_mean[:,i+1] + trend_damp*out[:,i] + corr * level_mean[:,i+1]

        return out.detach()

    @property
    def trend_scale(self):

        guide_trend_scale = self.dmm.scale_transform(self.dmm.guide_trend_scale)
        guide_corr = utils.sigmoid(self.dmm.guide_corr[:,0], min = -1.0, max = 1.0)
        level_scale = self.level_scale
        level_trend_cov = self.level_trend_cov

        out = torch.zeros( (self.dmm.latent_dim, self.dmm.num_years) )

        out[:,0] = torch.sqrt( guide_trend_scale[:,0]**2 + ( guide_corr * level_scale[:,0] )**2 )

        for i in range(self.dmm.num_years-1):
            corr = utils.sigmoid(self.dmm.guide_corr[:,i+1], min = -1.0, max = 1.0)
            trend_damp = utils.sigmoid(self.dmm.guide_trend_damp[:,i], min = 0.0, max = 1.0)
            out[:,i+1] = torch.sqrt( (guide_trend_scale[:,i+1])**2 + (trend_damp * out[:,i])**2 + (corr*level_scale[:,i+1])**2  + 2*corr*level_trend_cov[:,i+1])

        return out.detach()

    @property
    def dampening(self):
        return utils.sigmoid(self.dmm.latent_transition.dampening, min = 0.0, max = 1.0)

    @property
    def correlation(self):
        return utils.sigmoid(self.dmm.guide_corr, min = -1.0, max = 1.0)

    @property
    def guide_sample(self):
        (level, trend) = self.dmm.guide(self.data)
        return ( level.detach(), trend.detach() )
    
    def mortality_curve(self, time):
        return self.dmm.emitter.forward(self.level_loc[:,time]).detach()

    @property
    def factor_loadings(self):

        epsilon = 0.01

        #Bias term
        out = [self.dmm.emitter.forward(torch.zeros(self.dmm.latent_dim)).detach()]

        for i in range(self.dmm.latent_dim):
            z = torch.zeros(self.dmm.latent_dim)
            z[i] = epsilon

            out.append( (torch.log( self.dmm.emitter.forward(z).detach() ) - torch.log( self.dmm.emitter.forward( torch.zeros(self.dmm.latent_dim) )).detach())/epsilon )

        return out

    def smoothed_latent_mortality(self, age, time):
        return self.dmm.emitter.forward(self.level_loc[:,time]).detach()[age]

    def mortality(self, age, mc_samples = 1000):
        exposure = self.data[1]

        mean_mortality = np.zeros(self.num_years)
        std_mortality = np.zeros(self.num_years)

        level_loc = self.level_loc
        level_scale = self.level_scale

        for t in range(self.num_years):
            mortality = np.zeros(mc_samples)
            for i in range(mc_samples):
                x = torch.distributions.Normal(level_loc[:,t] , level_scale[:,t]).sample()
                intensity = self.dmm.emitter.forward(x)[age]
                mortality[i] = torch.distributions.Poisson(intensity * exposure[t,age]).sample() / exposure[t,age]
            mean_mortality[t] = np.mean(mortality)
            std_mortality[t] = np.std(mortality)
        
        return (mean_mortality, std_mortality)

    # def forecast_sim(self, forecast_length = 10, mc_samples = 1000, percentiles = [10, 50, 90]):

    #     #Distribution of last time in training set
    #     last_level_loc = self.level_loc[:,-1]
    #     last_trend_loc = self.trend_loc[:,-1]

    #     last_level_scale = self.level_scale[:,-1]
    #     last_trend_scale = self.trend_scale[:,-1]

    #     last_corr = self.level_trend_cov[:,-1]

    #     #Latent process standard deviation
    #     level_noise_scale = self.dmm.scale_transform( self.dmm.level_scale)
    #     trend_noise_scale = self.dmm.scale_transform( self.dmm.trend_scale)

    #     #Declare arrays to hold simulations
    #     level_forecast = torch.zeros((last_level_loc.size()[0],forecast_length, mc_samples))
    #     trend_forecast = torch.zeros((last_trend_loc.size()[0],forecast_length, mc_samples))

    #     for j in range(mc_samples):

    #         #Sample last time in training set
    #         last_level = torch.distributions.Normal( last_level_loc, last_level_scale ).sample()

    #         last_trend_mean = last_trend_loc + last_corr * last_trend_scale /last_level_scale * (last_level - last_level_loc)
    #         last_trend_std = last_trend_scale * torch.sqrt(1-last_corr*last_corr)

    #         last_trend = torch.distributions.Normal( last_trend_mean, last_trend_std ).sample()
            
    #         #Get mean of next time step
    #         (level_loc, trend_loc) = self.dmm.latent_transition.forward( last_level, last_trend )

    #         #Sample next time step
    #         level_forecast[:, 0, j] = torch.distributions.Normal( level_loc, level_noise_scale ).sample()
    #         trend_forecast[:, 0, j] = torch.distributions.Normal( trend_loc, trend_noise_scale ).sample()
            
    #         for i in range(1,forecast_length):
    #             (level_loc, trend_loc) = self.dmm.latent_transition.forward( level_forecast[:, i-1, j], trend_forecast[:, i-1, j] )
    #             level_forecast[:, i, j] = torch.distributions.Normal(level_loc , level_noise_scale ).sample()
    #             trend_forecast[:, i, j] = torch.distributions.Normal(trend_loc , trend_noise_scale ).sample()

    #     #Calculate mean and standard deviation of forecast
    #     level_mean_forecast = torch.zeros((last_level_loc.size()[0],forecast_length))
    #     level_scale_forecast = torch.zeros((last_level_loc.size()[0],forecast_length))

    #     for i in range(forecast_length):
    #         for j in range(last_level_loc.size()[0]):
    #             level_mean_forecast[j,i] = torch.mean(level_forecast[j, i, :])
    #             level_scale_forecast[j,i] = torch.std(level_forecast[j, i, :])

    #     #Calculate percentiles
    #     p_list = []

    #     for p in percentiles:
    #         level_perc_forecast = np.zeros((last_level_loc.size()[0],forecast_length))

    #         for i in range(forecast_length):
    #             for j in range(last_level_loc.size()[0]):
    #                 level_perc_forecast[j,i] = np.percentile(level_forecast[j, i, :].detach().numpy(), p)

    #     p_list.append( level_perc_forecast.copy() )

    #     return (level_mean_forecast.detach().numpy(),
    #             level_scale_forecast.detach().numpy(),
    #             p_list )

    def forecast_mortality_rate(self, age, forecast_length, mc_samples = 1000):

        exposure = self.data[1]

        (level_loc_forecast, level_scale_forecast) = self.forecast_factors(forecast_length)

        mean_mortality_rate = np.zeros(forecast_length)
        std_mortality_rate = np.zeros(forecast_length)

        for t in range(forecast_length):
            intensity = np.zeros(mc_samples)
            for i in range(mc_samples):
                factors = torch.distributions.Normal(level_loc_forecast[:,t] , level_scale_forecast[:,t]).sample()
                intensity[i] = self.dmm.emitter.forward(factors)[age]
            mean_mortality_rate[t] = np.mean(intensity)
            std_mortality_rate[t] = np.std(intensity)

        return (mean_mortality_rate, std_mortality_rate)


    def forecast_factors(self, forecast_length):

        #Distribution of last time in training set
        level_loc = self.level_loc[:,-1]
        trend_loc = self.trend_loc[:,-1]

        level_scale = self.level_scale[:,-1]
        trend_scale = self.trend_scale[:,-1]

        cov = self.level_trend_cov[:,-1]

        #Latent process standard deviation
        level_noise_scale = self.dmm.scale_transform(self.dmm.level_scale)
        trend_noise_scale = self.dmm.scale_transform(self.dmm.trend_scale)

        #Arrays to hold forecast
        level_forecast_loc = torch.zeros((level_loc.size()[0], forecast_length))
        level_forecast_scale = torch.zeros((trend_loc.size()[0], forecast_length))

        for t in range(forecast_length):
            (level_loc, trend_loc) = self.dmm.latent_transition.forward( level_loc, trend_loc )
            (level_scale, trend_scale, cov) = self.dmm.latent_transition.forward_cov( level_scale, trend_scale, cov, level_noise_scale, trend_noise_scale )

            level_forecast_loc[:,t] = level_loc
            level_forecast_scale[:,t] = level_scale
        
        return (level_forecast_loc.detach(), level_forecast_scale.detach())

    def evaluate(self, e, d, mc_samples = 1000):

        forecast_length = len(e)

        (fcast_mean, fcast_scale) = self.forecast_factors(forecast_length = forecast_length)

        log_score_t = np.zeros(forecast_length)
        log_score = np.zeros((mc_samples, self.num_ages))

        for t in range(forecast_length):
            for i in range(mc_samples):
                factor_sample = torch.distributions.Normal(fcast_mean[:,t] , fcast_scale[:,t] ).sample()
                intensity = self.dmm.emitter.forward(factor_sample).detach().numpy()
                for age in range(self.num_ages):
                    log_score[i,age] = poisson.logpmf(d[t,age], intensity[age] * e[t,age])
            log_score_t[t] = - np.sum( logsumexp(log_score, axis = 0) - np.log(mc_samples) )
        
        return log_score_t
        
        





