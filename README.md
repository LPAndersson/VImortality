# Forecasting mortality using variational inference

Implementation of [''Forecasting mortality using variational inference''](https://arxiv.org/abs/21xx.xxxxx).

## Dependencies

* NumPy
* SciPy
* PyTorch
* Pyro
* Pandas


To install Pyro, you can refer to https://pyro.ai

## Training

The script train.py trains models in parallell for the parameters specified in the file. The fitted model is saved in the folder fittedModels. 

## Forecast

The Jupyter notebook forecast.ipynb loads the trained model and produces the figures and results in the article.