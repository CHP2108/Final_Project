import streamlit as st  
import base64
import main_model
from main_model import *
import torch

st.header('Hello World')
menu=['1', '2','3']
st.sidebar.selectbox    ('RECOLOR IMAGE',menu)
# main_bg = "moneyback.jpeg"
# main_bg_ext = "jpg"

# side_bg = "Media/ab.jpg"
# side_bg_ext = "jpg"

# st.markdown(
# f"""
# <style>
# .reportview-container {{
#     background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
# }}
# .sidebar {{
#     background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
# }}
# </style>
# """,
# unsafe_allow_html=True)

model_resnet34 = torch.load('Models\Resnet34-vn.pt')
size=512
image=st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
if image is not None:
    image=Image.open(image)
    resized = image.resize((size,size))
    st.image(resized)
    gray=resized.convert('L')
    # to make it between -1 and 1
    img = transforms.ToTensor()(gray)[:1] * 2. - 1.
    model_resnet34.eval()
    with torch.no_grad():
        preds = model_resnet34.net_G(img.unsqueeze(0).to(device))
        img_pred = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
    st.image(img_pred)
    plt.figure(figsize=(15,8))
    ax = plt.subplot(1, 3, 1)
    ax.imshow(gray,cmap='gray')
    ax.axis("off")
    ax = plt.subplot(1, 3,2)
    ax.imshow(img_pred)
    ax.axis("off")
    ax = plt.subplot(1,3,3)
    ax.imshow(resized)
    ax.axis("off")
    plt.show()