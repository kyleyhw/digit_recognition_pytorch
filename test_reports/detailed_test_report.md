# Detailed Test Report

## Summary

- **Result:** OK
- **Tests Ran:** 8
- **Time:** 0.510s

---

## Test Results by Script

### `test_data_loader.py`

- `test_data_loaders_batch_size`: OK
- `test_data_normalization`: OK
- `test_data_shape_and_type`: OK
- `test_get_data_loaders_return_type`: OK

### `test_model.py`

- `test_forward_pass`: OK
- `test_model_creation`: OK

### `test_train.py`

- `test_test_function`: OK
- `test_train_function`: OK

---

## Full Test Output

```
test_data_loaders_batch_size (test_data_loader.TestDataLoader) ... ok
test_data_normalization (test_data_loader.TestDataLoader) ... ok
test_data_shape_and_type (test_data_loader.TestDataLoader) ... ok
test_get_data_loaders_return_type (test_data_loader.TestDataLoader) ... ok
test_forward_pass (test_model.TestModel) ... ok
test_model_creation (test_model.TestModel) ... ok
test_test_function (test_train.TestTrain) ... ok
test_train_function (test_train.TestTrain) ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.510s

OK

Test set: Average loss: 2.3230, Accuracy: 2/10 (20%)

Train Epoch: 1 [0/10 (0%)]	Loss: 2.364296
Train Epoch: 1 [4/10 (33%)]	Loss: 4.080024
Train Epoch: 1 [4/10 (67%)]	Loss: 1.946503
```