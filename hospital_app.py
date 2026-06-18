import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_mode():
  with open('hospital_model.pki', 'rb') as f:
    return pickle.load(f)



bundle = load_model()

# unpack the bundle into separate columns
model = bundle['model']
scaler = bundle['scaler']
features = bundle['features']
cols_to_scale = bundle['cols_to_scale']
dept_map_inv = bundle['dept_map_inv']
gender_map = bundle['gender_map']
temp_map = bundle['temp_map']
hr_map = bundle['hr_map']
dur_map = bundle['dur_map']
cc_map = bundle['cc_map']


DEPT_INFO = {
  'Respiratory Medicine' : {
    'icon': '💨',
    'color': '#e8ebdd'
    'desc': 'Specialises in conditions affecting the lungs and airways.',
    'step': ['Visit level 2, Wing B', 'Estimate wait: 6-7 mins', 'please wear mask or the hentai virus']
  }
  'Cardiology' : {
    'icon': '❤️',
    'color': '#ff0000'
    'desc': 'Specialises in conditions affecting the lungs and airways.',
    'step': ['Visit level 3, Wing A', 'Estimate wait: 12-23 mins', 'Bring any previous ECG reports']
  }
  'Gastroenterology' : {
    'icon': '🫃',
    'color': '#a0cc1b'
    'desc': 'Specialises in conditions affecting the lungs and airways.',
    'step': ['Visit level 2, Wing B', 'Estimate wait: 6-7 mins', 'please wear mask or the hentai virus']
  }
  'Cardiology' : {
    'icon': '❤️',
    'color': '#ff0000'
    'desc': 'Specialises in conditions affecting the lungs and airways.',
    'step': ['Visit level 3, Wing A', 'Estimate wait: 12-23 mins', 'Bring any previous ECG reports']
  }
}
67
