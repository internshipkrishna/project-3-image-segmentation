import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("segmentation_model.h5")

def preprocess_image(img):
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict(img):
    input_img = preprocess_image(img)
    mask = model.predict(input_img)[0]
    label = "flooded" if np.mean(mask) > 0.2 else "not flooded"
    return mask, label

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="numpy", label="Segmentation Mask"), gr.Textbox(label="Predicted Condition")],
    title="Disaster Image Segmentation",
    description="Upload a drone/satellite image to detect flood condition."
)

demo.launch()
