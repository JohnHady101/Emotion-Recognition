# Emotion Recognition

Image-based facial emotion recognition using an EfficientNetV2S model exported
to ONNX. The project includes a Streamlit interface for uploading an image and
a Python prediction module that can also be used directly from scripts.

## Features

- Predicts one of the seven FER2013 emotion classes:
	`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.
- Accepts JPG, JPEG, and PNG images through the Streamlit app.
- Runs inference locally with ONNX Runtime on the CPU.
- Reuses the ONNX session between predictions within the running process.
- Includes notebooks used for EfficientNetV2S training/export and a k6 load
	test script.

## Project Structure

```text
.
├── app/
│   ├── main.py                  # Streamlit application
│   └── predict.py               # Image preprocessing and ONNX inference
├── images/
│   └── PrivateTest_10131363.jpg # Example input image
├── notebooks/
│   ├── efficientnet.ipynb       # Training/export notebook
│   └── notebook116bf4db71.ipynb # Additional notebook
├── onnx_files/
│   └── best_effv2s.onnx         # Exported EfficientNetV2S model
├── requirements.txt             # Python dependencies
└── test.js                      # k6 HTTP load test
```

## Requirements

- Python 3.9 or newer
- pip
- A CPU that can run ONNX Runtime

The inference module also requires NumPy, Pillow, PyTorch, and Torchvision.
Install them alongside the pinned project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install numpy pillow torch torchvision
```

For a CUDA-enabled PyTorch installation, use the PyTorch installation command
appropriate for your CUDA version before installing `torchvision`.

## Run the Web App

From the project root:

```bash
source .venv/bin/activate
streamlit run app/main.py
```

Open the URL printed by Streamlit, upload a face image, and the predicted
emotion will be shown below the preview.

The application resolves the model relative to the repository root, so it
does not require changing the working directory or copying the model into
`app/`.

## Use the Predictor Directly

From the project root, run:

```bash
python -c "from app.predict import predict; print(predict('images/PrivateTest_10131363.jpg', 'onnx_files/best_effv2s.onnx'))"
```

Or use the module as a script after updating its example paths if needed:

```bash
python app/predict.py
```

The `predict` function accepts either an image path or a `PIL.Image.Image` and
returns the predicted class name:

```python
from app.predict import predict

emotion = predict(
		"images/PrivateTest_10131363.jpg",
		"onnx_files/best_effv2s.onnx",
)
print(emotion)
```

## Inference Contract

Input images are:

1. Converted to RGB.
2. Resized to `112 x 112` pixels.
3. Converted to floating-point values in the `[0, 1]` range.
4. Rearranged to a batch-first NHWC tensor with shape `(1, 112, 112, 3)`.

The ONNX model already includes its EfficientNetV2S preprocessing and final
softmax activation. The predictor therefore does not apply additional
normalization or a second softmax. The class-name order in
`app/predict.py` must remain aligned with the order used during training.

## Training and Model Export

The notebooks contain the FER2013 training and ONNX export workflow. After
exporting a replacement model, place it at:

```text
onnx_files/best_effv2s.onnx
```

Verify that the replacement model expects the same input shape and that its
output class order matches `CLASS_NAMES` in `app/predict.py`.

## Load Testing

`test.js` is a k6 script configured for 50 virtual users over 500 seconds:

```bash
k6 run test.js
```

Before running it, update the URL in `test.js` to the host and port where the
Streamlit app is reachable. The committed URL is an environment-specific
private IP and may not be accessible from another machine. The script measures
HTTP reachability and response behavior; it does not upload images or exercise
the prediction path.

## Troubleshooting

- **`streamlit: command not found`**: activate `.venv` or run
	`.venv/bin/streamlit run app/main.py`.
- **Model not found**: run the command from the project root and confirm that
	`onnx_files/best_effv2s.onnx` exists.
- **Import errors for Torchvision or Pillow**: install the additional inference
	dependencies shown above.
- **Unexpected labels**: confirm that the model's output order matches
	`CLASS_NAMES` in `app/predict.py`.

## License and Dataset

This repository does not currently include a license file. Review the terms of
the FER2013 dataset and any pretrained model components before redistribution
or commercial use.

