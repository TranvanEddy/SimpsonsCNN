import streamlit as st
from fastai.vision.all import load_learner, Path, torch, PILImage
import datetime

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("simpsons.csv", index_col=0)
data = data.iloc[:,:-1]

model = load_learner("simpsonsResNet34.pkl")

st.title("Reconnaissance de personnages?")
img = PILImage.create(r'les_simpson.jpg')
st.image(img, caption="LES SIMPSONS")
st.text("Cette application reconnait les personnages des Simpson.")
st.text("J'ai utilisé le modèle pre-entrainé resnet34.")
st.text("Mon modèle s'est entrainé sur 20933 images réparties en 42 classes")
st.write(data.head(42))
st.text("Le tableau ci-dessus indique le nombre d'images par personnages.")
img_upload = st.sidebar.file_uploader( "choisissez votre image", type=['jpg', 'png'])

if img_upload is not None:
    today = st.date_input("Aujourd'hui nous sommes le ", datetime.datetime.now())
    img_upload = PILImage.create(img_upload)
    st.image(img_upload)
    if st.button('prédiction'):
        pred=model.predict(img_upload)[0]
        st.balloons()
        st.success(f'Ce personnage semble être : {pred}')