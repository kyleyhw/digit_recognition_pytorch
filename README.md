# PyTorch Digit Recognizer

This project is a handwritten digit recognizer built using PyTorch. It utilizes a Convolutional Neural Network (CNN) to classify digits from the MNIST dataset. The project also includes a graphical user interface (GUI) that allows users to draw their own digits and see the model's predictions in real-time.

## Features

- **Convolutional Neural Network (CNN)**: A `Net` class defining the model architecture in `model.py`.
- **Training and Evaluation**: Scripts to train the model on the MNIST dataset and evaluate its performance.
- **Interactive GUI**: A user-friendly interface to draw digits and get predictions from the trained model.
- **Detailed Documentation**: In-depth explanation of the mathematical principles behind the code in the `/docs` directory.
- **Unit and Integration Tests**: A suite of tests to ensure the correctness of the data loading, model, and training functions.

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
│   ├───data_loader.md
│   ├───model.md
│   └───train.md
├───models/
│   └───mnist_cnn_subset_1200.pt
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

To train the model, run the `train.py` script. The trained model will be saved as `mnist_cnn.pt`.

```bash
python train.py
```

*Note: The current configuration trains on a subset of the data for faster execution. You can modify `train.py` to use the full dataset.*

### GUI

Once the model is trained, you can launch the GUI to test it with your own drawings.

```bash
python gui.py
```

## Testing

To run the test suite, use `pytest`:

```bash
pytest
```
