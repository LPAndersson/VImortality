# Forecasting mortality using variational inference

Implementation of [''Forecasting mortality using variational inference''](http://arxiv.org/abs/2305.15943).

## Dependencies

* NumPy
* SciPy
* PyTorch
* Pyro
* Pandas

To install Pyro, you can refer to https://pyro.ai

## Training

The script train.py trains models in parallell for the parameters specified in the file. The fitted model is saved in the folder trainedModels. Data necessary for training the model on Swedish data is provided in the data folder. For other countries the data is freely available at https://www.mortality.org 

## Forecast

The Jupyter notebooks in the folder scripts loads the trained model and produces the figures and results in the article