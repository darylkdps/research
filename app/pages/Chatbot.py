import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title='Chatbot',
    page_icon='📏',
    layout='wide',
    initial_sidebar_state='auto'  # auto, expanded, collapsed
    )

if 'C:\\' in str(Path.cwd()) or 'D:\\' in str(Path.cwd()):
    data_path = Path.cwd() / 'pages'
else:
    data_path = Path.cwd() / 'app' / 'pages'


st.title('Chat Bot')