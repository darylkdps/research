import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title='Research',
    page_icon='random',
    layout='wide',
    initial_sidebar_state='auto'
    )

st.title('Research')

# if 'D:' in str(Path.cwd()) or 'C:' in str(Path.cwd()):
#     st.markdown('_No token_')
# else:
#     st.markdown('_' + st.secrets['test_token'] + '_')


