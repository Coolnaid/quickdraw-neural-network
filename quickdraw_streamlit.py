import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from quickdraw_dataset import load_model

# load
@st.cache_resource
def get_model():
    model = load_model("model.npz")  # path must be right
    return model

# init
model = get_model()

# choose your doodles
class_names = ['bee', 'car', 'cat', 'dog', 'sailboat']

# streamlit
st.title("QuickDraw Doodle Recognizer")

# ui stuff
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#000000")
bg_color = st.sidebar.color_picker("Background color hex: ", "#FFFFFF")
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=255,
    width=255,
    drawing_mode="freedraw",
    point_display_radius=0,
    key="canvas",
)

# process
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8'))

    # resize to 28x28 and convert to grayscale
    resized_img = img.resize((28, 28)).convert("L")

    # display
    st.image(resized_img, caption="Resized Image (28x28)", width=150)

    # numpy
    img_array = np.array(resized_img).flatten() / 255.0  # normalize

    if st.button("Predict"):
        prediction = model.forward(img_array.reshape(1, -1))
        predicted_class = np.argmax(prediction, axis=1)[0]

        st.write(f"Predicted Doodle: **{class_names[predicted_class]}**")
