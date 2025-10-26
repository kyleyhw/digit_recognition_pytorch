## `tests/test_data_loader.py`

**Runtime:** 0.26s (sum of relevant durations)

### `test_get_data_loaders_return_type`
- **What:** Unit test for the `get_data_loaders()` function's return type.
- **Why:** To ensure the function returns `DataLoader` objects as expected.
- **Specifically Tested:** Verified that both `train_loader` and `test_loader` are instances of `torch.utils.data.DataLoader`.
- **Test Data Rationale:** Default parameters were used as the batch size does not affect the return type.

### `test_data_loaders_batch_size`
- **What:** Unit test for the `batch_size` argument of `get_data_loaders()`.
- **Why:** To confirm that the `DataLoader` yields batches of the specified size.
- **Specifically Tested:** Checked if the first dimension of the image tensor from the `train_loader` matches the `batch_size` parameter.
- **Test Data Rationale:** A `batch_size` of 32 was chosen as a standard, representative batch size.

### `test_data_shape_and_type`
- **What:** Unit test for the shape and data type of the tensors produced by the `DataLoader`.
- **Why:** To ensure the data is in the correct format for the model's input layer.
- **Specifically Tested:**
    - Image tensor shape is `[1, 1, 28, 28]` for a single sample.
    - Label tensor shape is `[1]` for a single sample.
    - Image tensor type is `torch.FloatTensor`.
    - Label tensor type is `torch.LongTensor`.
- **Test Data Rationale:** A `batch_size` of 1 was used to simplify the checking of individual tensor shapes.

### `test_data_normalization`
- **What:** Unit test for the data normalization.
- **Why:** To verify that the image data is correctly normalized to the range `[-1, 1]`.
- **Specifically Tested:** Checked that all pixel values in a batch of images are within the `[-1.0, 1.0]` range.
- **Test Data Rationale:** A batch of 128 images was used to get a reasonable sample size for checking the normalization.