# PyTorch Digit Recognizer

This project is a handwritten digit recognizer built using PyTorch. It utilizes a Convolutional Neural Network (CNN) to classify digits from the MNIST dataset. The project also includes a graphical user interface (GUI) that allows users to draw their own digits and see the model's predictions in real-time.

## Features

- **Convolutional Neural Network (CNN)**: A `Net` class defining the model architecture in `model.py`.
- **Training and Evaluation**: Scripts to train the model on the MNIST dataset and evaluate its performance.
- **Interactive GUI**: A user-friendly interface to draw digits and get predictions from the trained model.
- **Detailed Documentation**: In-depth explanation of the mathematical principles behind the code.
- **Unit and Integration Tests**: A suite of tests to ensure the correctness of the data loading, model, and training functions.

## Documentation

For a complete overview of the project's documentation, please see the **[Documentation Index](docs/index.md)**.

## Project Structure

```
├───.gitignore
├───data_loader.py
├───environment.yml
├───model.py
├───train.py
├───gui.py
├───README.md
├───__pycache__/
├───docs/
│   ├───index.md
│   ├───data_loader.md
│   ├───model.md
│   └───train.md
├───models/
│   ├───mnist_cnn_subset_1200.pt
│   └───mnist_cnn_subset_12000.pt
├───test_reports/
│   ├───test_data_loader_report.md
│   ├───test_model_report.md
│   └───test_train_report.md
└───tests/
    ├───test_data_loader.py
    ├───test_model.py
    ├───test_train.py
    └───__pycache__/
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kyleyhw/digit_recognition_pytorch.git
    cd digit_recognition_pytorch
    ```

2.  **Create the Conda environment:**
    This project uses a Conda environment to manage dependencies. Create the environment using the provided `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate digit-recognition-pytorch
    ```

## Usage

### Training

The `train.py` script is configured to train on a subset of the data for faster execution. You can modify the `main` function in `train.py` to change the subset size and number of epochs. The trained model will be saved to the project root and should be moved to the `models/` directory.

```bash
python train.py
```

### GUI

The GUI uses the latest trained model. Make sure the `gui.py` script is pointing to the correct model file in the `models/` directory.

```bash
python gui.py
```

## Testing

To run the test suite, use `pytest`:

```bash
pytest
```