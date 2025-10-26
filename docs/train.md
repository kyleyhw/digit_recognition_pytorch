# `train.py` Documentation

This script is the main entry point for training and evaluating the MNIST digit classification model.

## Functions

### `train(model, train_loader, optimizer, epoch, log_interval=10)`

This function handles the training process for one epoch.

#### Loss Function

The training process uses the **Negative Log-Likelihood Loss (`nll_loss`)**. This loss function is useful to train a classification problem with C classes. The loss for a single sample is given by:

```
loss(x, class) = -x[class]
```

where `x` is the output of the `log_softmax` layer and `class` is the target class.

### `test(model, test_loader)`

This function evaluates the model's performance on the test dataset.

### `main()`

This is the main function that orchestrates the training and evaluation process.

#### Optimizer

The `Adadelta` optimizer is used. Adadelta is a stochastic gradient descent method that is based on adaptive learning rates for each parameter. It does not require a learning rate to be set.

#### Learning Rate Scheduler

The `StepLR` scheduler is used to decay the learning rate of the optimizer by a factor of `gamma` every `step_size` epochs.

## Usage

To start the training process, run the script from the command line:

```bash
python train.py
```