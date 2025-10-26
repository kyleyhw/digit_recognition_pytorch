# `train.py` Documentation

This script is the main entry point for training and evaluating the MNIST digit classification model.

## Core Concepts

### Epochs and Batches

-   **Epoch**: One complete pass through the entire training dataset.
-   **Batch**: A subset of the training dataset. Instead of processing the entire dataset at once, the data is split into smaller batches. This is more memory-efficient and allows the model to be updated more frequently.

### Loss Function: Negative Log-Likelihood Loss (`nll_loss`)

The loss function measures how well the model's prediction matches the true label. For a classification problem, we use the **Negative Log-Likelihood Loss**. This loss function is used in combination with a `log_softmax` output layer.

For a single prediction, the model outputs a vector of log-probabilities $z = (z_1, ..., z_K)$, where $z_i = \log(p_i)$ and $p_i$ is the probability of the input being class $i$. If the true class is $c$, the loss is simply the negative of the log-probability for that class:

$$
\text{Loss} = -z_c = -\log(p_c) 
$$

By minimizing this loss, we are maximizing the log-probability of the correct class. This is the principle of **Maximum Likelihood Estimation**.

#### Rationale for NLL Loss

-   **Suitability for Classification**: NLL Loss is a standard choice for multi-class classification problems, especially when the model's output layer provides log-probabilities (as `log_softmax` does).
-   **Numerical Stability**: Using `log_softmax` directly with `NLLLoss` is numerically more stable than computing `softmax` and then taking the logarithm, preventing potential underflow issues with very small probabilities.
-   **Direct Probability Maximization**: Minimizing NLL Loss is equivalent to maximizing the likelihood of the observed data under the model, which is a statistically sound approach to training classifiers.

### Backpropagation

The `loss.backward()` call initiates the **backpropagation** algorithm. This is the core mechanism for training neural networks. It calculates the gradient of the loss function with respect to each of the model's parameters (weights and biases). This is done by applying the chain rule of calculus, starting from the final layer and working backwards through the network.

For a parameter $w$, the gradient is $\frac{\partial \text{Loss}}{\partial w}$.

### Optimizer: `Adadelta`

The optimizer's role is to update the model's parameters using the gradients calculated by backpropagation, in order to minimize the loss function. `Adadelta` is an adaptive learning rate method.

Instead of a fixed learning rate, Adadelta adapts the learning rate for each parameter based on the history of gradients. The update rule for a parameter $\theta$ at timestep $t$ is:

$$ 
\Delta \theta_t = - \frac{\text{RMS}[\Delta \theta]_{t-1}}{\text{RMS}[g]_t} g_t 
$$

Where:
- $g_t$ is the gradient at time $t$.
- $\text{RMS}[\cdot]$ is the root mean square.

This method has the advantage of not requiring a default learning rate.

#### Rationale for Adadelta Optimizer

-   **Adaptive Learning Rate**: Adadelta automatically adapts the learning rates for each parameter, eliminating the need for manual tuning of a global learning rate. This can significantly simplify hyperparameter optimization.
-   **Robustness**: It is generally robust to noisy gradients and can perform well across a variety of tasks without extensive tuning.
-   **No Learning Rate Decay Needed (Implicitly Handled)**: Unlike optimizers with a fixed learning rate, Adadelta's adaptive nature often means it handles learning rate decay implicitly, though a scheduler can still be used for further fine-tuning.

### Learning Rate Scheduler: `StepLR`

A learning rate scheduler adjusts the learning rate during training. The `StepLR` scheduler decays the learning rate by a factor of `gamma` every `step_size` epochs.

The learning rate at epoch `e` is given by:

$$ 
\text{lr}_e = \text{lr}_0 \times \gamma^{\lfloor e / \text{step\_size} \rfloor} 
$$

This helps to make larger updates at the beginning of training and smaller, more fine-tuning updates as training progresses.

#### Rationale for StepLR Scheduler and Parameters (gamma=0.7)

-   **Controlled Decay**: `StepLR` provides a simple yet effective way to reduce the learning rate at predefined intervals. This helps the model to make larger steps in the parameter space early in training (when it's far from the optimum) and then take smaller, more precise steps as it approaches convergence.
-   **Gamma (0.7)**: A common decay factor. Reducing the learning rate by 30% every epoch (as configured in `main()`) helps prevent oscillations around the minimum and allows for finer adjustments to the model weights in later epochs.

#### Rationale for Learning Rate (1.0)

-   **Adadelta's Nature**: For Adadelta, the initial learning rate (here 1.0) is often set to a default value as the optimizer itself adapts the per-parameter learning rates. It's less sensitive to the initial global learning rate compared to optimizers like SGD.

#### Rationale for Epochs (14)

-   **Empirical Choice**: The number of epochs (14 for the full dataset) is typically determined empirically. It's chosen to allow the model sufficient time to converge without overfitting. This value was found to provide good performance on the MNIST dataset within a reasonable training time.

#### Rationale for Log Interval (10)

-   **Monitoring Progress**: The `log_interval` (10 batches) determines how frequently training progress (loss) is printed to the console. A value of 10 provides a good balance between seeing frequent updates and not flooding the console with too much information.

## Functions

-   `train()`: Handles the training loop for one epoch.
-   `evaluate()`: Evaluates the model on the test set.
-   `save_checkpoint()`: Saves the current state of the model and optimizer.
-   `load_checkpoint()`: Loads a saved state to resume training.
-   `main()`: Orchestrates the entire process, including checkpointing.

## Checkpointing

Checkpointing allows for saving the state of the training process at regular intervals. This is crucial for long training runs, as it enables resuming training from the last saved point in case of interruptions or for experimenting with different parameters without restarting from scratch.

### Naming Convention for Checkpoints

To support multiple training instances (e.g., training on different data subsets) simultaneously, a unique `run_id` is used. Checkpoints are saved with the following convention:

`checkpoints/checkpoint_<run_id>.pt`

This ensures that each training run has its own set of checkpoints, preventing conflicts and allowing for easy management of different experiments.

### Resuming Training

When `main()` is called, it first checks for an existing checkpoint file (`checkpoint_<run_id>.pt`) in the `checkpoints/` directory. If found, the model and optimizer states are loaded, and training resumes from the epoch saved in the checkpoint. Otherwise, training starts from epoch 1.

## Usage

To start the training process, run the script:

```bash
python train.py
```

*Note: The `main` function in `train.py` currently uses a subset of the MNIST dataset for faster training. You can modify the `range()` arguments for `train_dataset` and `test_dataset` to adjust the subset size, or remove these lines entirely to train on the full dataset.*
