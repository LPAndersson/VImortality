import sys
import importlib
sys.path.insert(0, '../')

import mortalityForecast as mf

import multiprocessing

param_tuple = ({
    'country' : mf.data_loader.SWEDEN,
    'first_year_train' : 1952,
    'last_year_train' : 2011,
    'first_year_test' : 2012,
    'last_year_test' : 2021,
    'sex' : "Male", # or Female, Both
    'max_age' : 100,
    'nn_layers' : 2,
    'latent_dim' : 2
},)


def train(param):

    exposure_all, deaths_all = mf.load_data(param, train = True, test = True)
    exposure_train, deaths_train = mf.load_data(param, train = True, test = False)
    exposure_test, deaths_test = mf.load_data(param, train = False, test = True)

    model = mf.Mortality(param)

    file_name = "../trainedModels/test_%s_%d_%d_%s_%d_%d_%d" %(param['country'].name, param['first_year_train'], param['last_year_train'], param['sex'], param['max_age'], param['nn_layers'], param['latent_dim'])

    model = model.fit(exposure = exposure_train, 
                      deaths = deaths_train, 
                      num_steps = 20000, 
                      log_freq = 100, 
                      lr = 0.01,
                      checkpoint_freq = 100,
                      lr_decay = 0.9999,
                      save_file = file_name)


if __name__ == '__main__':
    processes = []
    for param in param_tuple:
        process = multiprocessing.Process(target=train, 
                                        args=(param,))
        processes.append(process)
        process.start()

    for proc in processes:
        proc.join()