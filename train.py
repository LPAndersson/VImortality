from mortalityForecast import mortalityForecast as mf
from mortalityForecast import mortality_data_loader as data_loader
from mortalityForecast import utils

import matplotlib.pyplot as plt
import numpy as np

import multiprocessing

param_tuple = ({
    'country' : data_loader.USA,
    'first_year_train' : 1933,
    'last_year_train' : 2000,
    'first_year_test' : 2001,
    'last_year_test' : 2016,
    'sex' : "Male", # or Female, Both
    'max_age' : 100,
    'nn_layers' : 2,
    'latent_dim' : 1
},)

param_tuple = param_tuple + ({
    'country' : data_loader.USA,
    'first_year_train' : 1933,
    'last_year_train' : 2000,
    'first_year_test' : 2001,
    'last_year_test' : 2016,
    'sex' : "Male", # or Female, Both
    'max_age' : 100,
    'nn_layers' : 2,
    'latent_dim' : 2
},)

param_tuple = param_tuple + ({
    'country' : data_loader.USA,
    'first_year_train' : 1933,
    'last_year_train' : 2000,
    'first_year_test' : 2001,
    'last_year_test' : 2016,
    'sex' : "Male", # or Female, Both
    'max_age' : 100,
    'nn_layers' : 2,
    'latent_dim' : 3
},)

param_tuple = param_tuple + ({
    'country' : data_loader.USA,
    'first_year_train' : 1933,
    'last_year_train' : 2000,
    'first_year_test' : 2001,
    'last_year_test' : 2016,
    'sex' : "Male", # or Female, Both
    'max_age' : 100,
    'nn_layers' : 2,
    'latent_dim' :4
},)

param_tuple = param_tuple + ({
    'country' : data_loader.USA,
    'first_year_train' : 1933,
    'last_year_train' : 2000,
    'first_year_test' : 2001,
    'last_year_test' : 2016,
    'sex' : "Male", # or Female, Both
    'max_age' : 100,
    'nn_layers' : 2,
    'latent_dim' :5
},)

param_tuple = param_tuple + ({
    'country' : data_loader.USA,
    'first_year_train' : 1933,
    'last_year_train' : 2000,
    'first_year_test' : 2001,
    'last_year_test' : 2016,
    'sex' : "Male", # or Female, Both
    'max_age' : 100,
    'nn_layers' : 2,
    'latent_dim' :6
},)

param_tuple = param_tuple + ({
    'country' : data_loader.USA,
    'first_year_train' : 1933,
    'last_year_train' : 2000,
    'first_year_test' : 2001,
    'last_year_test' : 2016,
    'sex' : "Female", # or Female, Both
    'max_age' : 100,
    'nn_layers' : 2,
    'latent_dim' :3
},)

param_tuple = param_tuple + ({
    'country' : data_loader.USA,
    'first_year_train' : 1933,
    'last_year_train' : 2000,
    'first_year_test' : 2001,
    'last_year_test' : 2016,
    'sex' : "Both", # or Female, Both
    'max_age' : 100,
    'nn_layers' : 2,
    'latent_dim' :3
},)

def train(param):

    exposure_all, deaths_all = data_loader.load_data(param, train = True, test = True)
    exposure_train, deaths_train = data_loader.load_data(param, train = True, test = False)
    exposure_test, deaths_test = data_loader.load_data(param, train = False, test = True)

    model = mf.Mortality(param, cuda = False)

    file_name = "trainedModels/%s_%d_%d_%s_%d_%d_%d" %(param['country'].name, param['first_year_train'], param['last_year_train'], param['sex'], param['max_age'], param['nn_layers'], param['latent_dim'])

    model = model.fit(exposure = exposure_train, deaths = deaths_train, num_steps = 10000, log_freq = 10, checkpoint_freq = 100, save_file = file_name)


if __name__ == '__main__':
    processes = []
    for param in param_tuple:
        process = multiprocessing.Process(target=train, 
                                        args=(param,))
        processes.append(process)
        process.start()

    for proc in processes:
        proc.join()