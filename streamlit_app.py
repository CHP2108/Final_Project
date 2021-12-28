import streamlit as st  
import base64
import main_model
from main_model import *
import torch

st.sidebar.image('Media\ReColor.png')
st.sidebar.write('Application to quickly convert black-white to color image.')
picksize=st.sidebar.slider('Pick size:',256,800)
st.sidebar.header('GITHUB:')
st.sidebar.write('https://github.com/CHP2108/Recolor-Image')    
st.header('RECOLOR IMAGE ')
st.image('Media/anh-bia-facebook-thac-nuoc-dep.jpg')
model_resnet34 = torch.load('Models\Resnet34-vn_tuning.pt')
cl1,cl2=st.columns(2)
with cl1:
    st.write("Recolor black & white convert to colorful images. ")
    st.write("Upload your image and choose size to use. ")
with cl2:
    image=st.file_uploader("Upload Your Images (< 200 MB)", type=["png","jpg","jpeg"])
if image is not None:
    image=Image.open(image)
    if image.size[0] > image.size[1]:
        if image.size[0] > picksize:
            maxsize =picksize
            minsize =int(maxsize*(image.size[1]/image.size[0]))
            resized = image.resize((maxsize,minsize))
        else: resized = image.resize(image.size)
    else:
        if image.size[1] > picksize:
            maxsize =picksize
            minsize =int(maxsize*(image.size[0]/image.size[1]))
            resized = image.resize((minsize,maxsize))
        else: resized = image.resize(image.size)
    gray=resized.convert('L')
    # to make it between -1 and 1
    img = transforms.ToTensor()(gray)[:1] * 2. - 1.
    model_resnet34.eval()
    with torch.no_grad():
        preds = model_resnet34.net_G(img.unsqueeze(0).to(device))
        img_pred = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
    col1,col2=st.columns(2)
    with col1: 
        st.write('Black-white image: ')
        st.image(gray)
    with col2:
        st.write('Recolor image:')
        st.image(img_pred)
    show_original= st.checkbox('show original images')
    if show_original:
        st.image(image)
        
