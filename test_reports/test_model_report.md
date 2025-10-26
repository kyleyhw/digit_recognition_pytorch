## `tests/test_model.py`

**Runtime:** 0.02s (sum of relevant durations)

### `test_model_creation`
- **What:** Unit test for the `Net()` model instantiation.
- **Why:** To ensure the model object can be created without errors.
- **Specifically Tested:** Verified that the created object is an instance of the `Net` class.
- **Test Data Rationale:** Not applicable.

### `test_forward_pass`
- **What:** Unit test for the model's `forward()` method.
- **Why:** To ensure a forward pass can be completed without errors and produces an output of the correct shape.
- **Specifically Tested:** Passed a random tensor of shape `[1, 1, 28, 28]` through the model and checked that the output tensor has the shape `[1, 10]`.
- **Test Data Rationale:** A random tensor with the same dimensions as a single MNIST image (`1, 1, 28, 28`) was used to simulate a valid input to the model.