import time
import streamlit as st

with st.spinner('Wait for it...'):
    time.sleep(1)
st.success('작업중입니다.')
