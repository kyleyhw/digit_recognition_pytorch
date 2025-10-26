# `gui.py` Documentation

This document explains the functionality of the `gui.py` script, focusing on the interactive digit drawing and real-time prediction features, particularly the image centering preprocessing.

## Image Centering: `center_image(self, img_array)`

### Problem Statement

Neural networks, especially those trained on datasets like MNIST, perform optimally when the input digits are consistently positioned and scaled. If a user draws a digit off-center or too small/large on the canvas, the model's prediction accuracy can significantly decrease. To combat this, a preprocessing step is implemented to automatically center and scale the drawn digit.

### Mathematical and Algorithmic Details

The `center_image` method takes the raw 28x28 pixel grid (represented as a NumPy array) from the drawing canvas and transforms it into a centered, scaled 28x28 input suitable for the neural network. The process involves several steps:

1.  **Bounding Box Detection**: The first step is to identify the minimal rectangular region that encloses all the drawn pixels. This is achieved by:
    -   `rows = np.any(img_array, axis=1)`: This creates a boolean array where `True` indicates a row containing at least one drawn pixel.
    -   `cols = np.any(img_array, axis=0)`: Similarly, this identifies columns with drawn pixels.
    -   `rmin, rmax = np.where(rows)[0][[0, -1]]`: Finds the indices of the first and last `True` values in the `rows` array, defining the top and bottom boundaries of the bounding box.
    -   `cmin, cmax = np.where(cols)[0][[0, -1]]`: Finds the left and right boundaries of the bounding box.
    -   A check is included for an empty canvas, returning an all-zeros array if no pixels are drawn.

2.  **Cropping**: The digit is then extracted from the original 28x28 grid using these bounding box coordinates:
    ```python
    cropped = img_array[rmin:rmax+1, cmin:cmax+1]
    ```

3.  **Aspect Ratio Scaling to 20x20**: The MNIST dataset typically contains digits scaled to fit within a 20x20 pixel box, which is then placed in the center of a 28x28 canvas. To mimic this, the cropped digit is scaled to fit a 20x20 square while preserving its aspect ratio:
    -   The `rows` and `cols` of the `cropped` image are determined.
    -   A `factor` is calculated based on whether the height or width is larger, ensuring the larger dimension scales down to 20 pixels.
    -   `PIL.Image.fromarray` converts the NumPy array to a PIL Image.
    -   `img.resize((cols, rows), Image.LANCZOS)` resizes the image. `Image.LANCZOS` is a high-quality downsampling filter.
    -   The resized image is converted back to a NumPy array and normalized.

4.  **Centering in 28x28 Canvas**: The scaled 20x20 (or smaller, if aspect ratio was preserved) digit is then placed into the center of a new 28x28 NumPy array:
    -   A `centered_array` of zeros (28x28) is created.
    -   `row_start` and `col_start` are calculated to position the `resized_array` in the exact center of the `centered_array`.
    -   The `resized_array` is then copied into the `centered_array` at these calculated offsets.

### Integration with `predict_realtime`

The `predict_realtime` method now calls `self.center_image(self.grid)` at the beginning of its execution. This ensures that the model always receives a consistently centered and scaled digit, regardless of how the user draws it on the canvas. The output of `center_image` is then converted to a PyTorch tensor and passed to the neural network for prediction.
