import streamlit as st
import mmcv
import os
import numpy as np
from PIL import Image
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from pathlib import Path
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Table Detection from Images")

config_file = '/content/CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
checkpoint_file = '/content/epoch_36.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    directory = "tempDir"
    path = os.path.join(os.getcwd(), directory)
    p = Path(path)
    if not p.exists():
        os.mkdir(p)
    with open(os.path.join(path, uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer()) 
    file_loc = os.path.join(path, uploaded_file.name)
    result = inference_detector(model, file_loc)
    st.pyplot(show_result_pyplot(file_loc, result,('Bordered', 'cell', 'Borderless'), score_thr=0.85))
    # st.pyplot(show_result_pyplot(file_loc, result,('Bordered', 'Borderless'), score_thr=0.85))