# Forecasting mortality using variational inference

Implementation of [''Forecasting mortality using variational inference''](https://arxiv.org/abs/21xx.xxxxx).

## Dependencies

* NumPy
* SciPy
* PyTorch
* pyro
* pandas


To install TensorFlow, you can refer to https://pyro.ai

## Usage

The following example usage shows how to train and test a TPA-LSTM model on MuseData with settings used in this work.

### Training

```
$ python main.py --mode train \
    --attention_len 16 \
    --batch_size 32 \
    --data_set muse \
    --dropout 0.2 \
    --learning_rate 1e-5 \
    --model_dir ./models/model \
    --num_epochs 40 \
    --num_layers 3 \
    --num_units 338
```

### Testing

```
$ python main.py --mode test \
    --attention_len 16 \
    --batch_size 32 \
    --data_set muse \
    --dropout 0.2 \
    --learning_rate 1e-5 \
    --model_dir ./models/model \
    --num_epochs 40 \
    --num_layers 3 \
    --num_units 338
```