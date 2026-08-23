import streamlit as st
from pathlib import Path
from PIL import Image


from predict import predict

st.set_page_config(page_title="Emotion Recognition", page_icon="📷")
st.title("📷 Emotion Recognition")
st.write("Upload a picture to predict the facial emotion.")



uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    st.image(
        image,
        caption="Selected image",
        use_container_width=True
    )

    model_path = (
        Path(__file__).resolve().parent.parent
        / "onnx_files"
        / "best_effv2s.onnx"
    )

    try:
        emotion = predict(image, str(model_path))
    except Exception as error:
        st.error(f"Could not analyze the image: {error}")
    else:
        st.success(f"Predicted emotion: {emotion}")


if __name__ == "__main__":
    model_path = Path(__file__).resolve().parent.parent / "onnx_files" / "best_effv2s.onnx"
    print(f"Model path: {model_path}")
