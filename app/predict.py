"""
Predict the facial emotion in an image using your exported FER2013
EfficientNetV2S ONNX model.

Preprocessing mirrors the training notebook exactly:
- resize to 112x112, RGB
- scale to [0, 1] (no extra mean/std - EfficientNetV2S's own preprocessing
  is already baked into the ONNX graph, so nothing else is applied here)
- channels-last (NHWC), since the model is Keras/TensorFlow-derived

The model's final layer already has activation='softmax', so the ONNX
output is already a probability distribution - no softmax is reapplied.

    pip install onnxruntime torch torchvision pillow
"""
from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

IMG_SIZE = 112  # must match training

# Standard FER2013 (msambare/fer2013) class order - this is sorted()
# alphabetically, matching how the notebook builds `emotion_classes`.
# Double check this against what your training run printed; if it differs,
# just edit this list to match.
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # PIL (H,W,C) 0-255 -> float tensor (C,H,W) in [0, 1]
])

_session = None  # loaded once, reused across calls


def _get_session(onnx_path: str) -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return _session


def predict(image: Union[str, Path, Image.Image], onnx_path: str = "best_effv2s.onnx") -> str:
    """Take an image (path or PIL Image), return the predicted emotion label."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = image.convert("RGB")

    tensor = _transform(image)                                  # (3, 112, 112), [0, 1]
    array = tensor.permute(1, 2, 0).numpy()                      # -> (112, 112, 3), channels-last
    array = np.expand_dims(array, axis=0).astype(np.float32)      # -> (1, 112, 112, 3)

    session = _get_session(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    output = session.run([output_name], {input_name: array})[0]
    probs = np.squeeze(output)  # already softmax probabilities

    predicted_index = int(np.argmax(probs))
    return CLASS_NAMES[predicted_index]


if __name__ == "__main__":
    import sys

    image_path = "/opt/dlami/nvme/Emotion-Recognition/images/PrivateTest_10131363.jpg"
    onnx_path = "/opt/dlami/nvme/Emotion-Recognition/onnx_files/best_effv2s.onnx"
    print(predict(image_path, onnx_path))