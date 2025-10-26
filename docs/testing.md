# Testing Documentation

This document provides a comprehensive overview of the project's test suite, including its structure, purpose, and how to interpret the generated test reports.

## Purpose of the Test Suite

The test suite is designed to ensure the correctness, reliability, and maintainability of the `digit_recognition_pytorch` project. It covers various aspects of the codebase:

-   **Unit Tests**: Verify that individual functions and components (e.g., data loading, model layers) work as expected in isolation.
-   **Integration Tests**: Confirm that different modules and components interact correctly when combined (e.g., the training loop, model's forward pass).

## Test Organization

Tests are organized by the module they cover, residing in the `tests/` directory. Each Python file in `tests/` corresponds to a main script in the project root:

-   `tests/test_data_loader.py`: Contains tests for the `data_loader.py` script.
-   `tests/test_model.py`: Contains tests for the `model.py` script.
-   `tests/test_train.py`: Contains tests for the `train.py` script.

All test files use the `unittest` framework, with test methods prefixed with `test_`.

## How to Run Tests

To execute the entire test suite, navigate to the project's root directory in your activated Conda environment and run `pytest`:

```bash
pytest
```

`pytest` will automatically discover and run all tests within the `tests/` directory.

## Interpreting Test Reports

After running `pytest`, detailed test reports are generated and stored in the `test_reports/` directory. These reports adhere to specific standards to provide clear and actionable feedback.

Each individual test report (e.g., `test_reports/test_data_loader_report.md`) includes:

1.  **Runtime**: The execution time of the tests within that module.
2.  **What was done**: A description of the unit or integration test performed.
3.  **Why it was done**: The rationale behind testing that specific functionality (e.g., to verify correct handling of boundary conditions).
4.  **What was specifically tested**: Details about the inputs and expected outputs or behaviors.
5.  **Test Data Rationale**: Justification for the choice of specific testing input data.
6.  **Failure Handling**: (If applicable) Documentation of any test failures, the error encountered, and the fix implemented.

### Accessing Individual Test Reports

-   **[Data Loader Test Report](test_reports/test_data_loader_report.md)**
-   **[Model Test Report](test_reports/test_model_report.md)**
-   **[Train Test Report](test_reports/test_train_report.md)**
