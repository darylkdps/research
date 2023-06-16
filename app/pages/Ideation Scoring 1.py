import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import re
import spacy

st.set_page_config(
    page_title='Ideation Scoring 1',
    page_icon='📏',
    layout='wide',
    initial_sidebar_state='auto'
    )

caption_placeholder = st.empty()
st.title('Ideation Scoring 1')

@st.cache_resource
def load_model():
    spacy.require_cpu()
    return spacy.load('en_core_web_lg')
nlp = load_model()

@st.cache_resource
def clean_text(text: str) -> str:
    '''Clean text by remapping specified characters, then removing extraneous spaces.'''
    # Characters mapping table
    dict_map = {
        '“':  '"',  # LEFT DOUBLE QUOTATION MARK: QUOTATION MARK
        '”':  '"',  # RIGHT DOUBLE QUOTATION MARK: QUOTATION MARK
        '‘':  "'",  # LEFT SINGLE QUOTATION MARK: APOSTROPHE
        '’':  "'",  # RIGHT SINGLE QUOTATION MARK: APOSTROPHE
        '–':  '-',  # EN DASH: HYPHEN-MINUS
        '\t': ' ',  # Horizontal tab
        '\n': ' ',  # Line feed
        '\v': ' ',  # Vertical tab
        '\f': ' ',  # Form feed
        '\r': ' ',  # Carriage return
    }

    # Map chacters
    mapped_text = text.translate(str.maketrans(dict_map))

    # Remove extraneous spaces
    extraneous_spaces_stripped = re.compile(' {1,}').sub(' ', mapped_text)

    return extraneous_spaces_stripped

@st.cache_resource
def tokenise(text: str) -> list:
    '''Convert string to spacy.Language object and get valid and lemmatised tokens.'''
    open_class_pos = ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']  # Excluded 'INTJ'

    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if (
            token.is_alpha and
            token.ent_type_ == '' and
            token.pos_ in open_class_pos and
            not token.is_stop and
            token.has_vector and
            len(token) > 1
        )
    ]

    return tokens

metadata1 = {key:value for key, value in nlp.meta.items() if key in {'lang', 'name', 'version', 'vectors'}}
metadata2 = f"*spaCy={spacy.__version__}, {metadata1['lang']}_{metadata1['name']}={metadata1['version']} ({metadata1['vectors']['vectors']} vectors with {metadata1['vectors']['width']} dimensions)*"
caption_placeholder.caption(metadata2)

default_title_words = ''
default_content_words = ''

sl_title_words = st.text_input(
    label='Title:',
    placeholder='Ideation',
    value=default_title_words,
    )

sl_content_words = st.text_input(
    label='Content:',
    placeholder='I had to ideate stuff during Design and Technology.',
    value=default_content_words,
    )

title_tokenised = tokenise(clean_text(sl_title_words))
content_tokenised = tokenise(clean_text(sl_content_words))

data = [(title_token, content_token, np.round(nlp.vocab[title_token].similarity(nlp.vocab[content_token]), 3)) for title_token in title_tokenised for content_token in content_tokenised]
df = pd.DataFrame(data=data, columns=['Title Word', 'Content Word', 'Cosine Similarity'])

column_config = {
    "Title Word": st.column_config.TextColumn(
        label=None,
        width=None,
        help="The title",
        disabled=True,
        required=None,
        default=None,
        max_chars=None,
        validate=None
    ),
    "Content Word": st.column_config.TextColumn(
        label=None,
        width=None,
        help="The content",
        disabled=True,
        required=None,
        default=None,
        max_chars=None,
        validate=None
    ),
    "Cosine Similarity": st.column_config.NumberColumn(
        label=None,
        width=None,
        help="The cosine similarity",
        disabled=True,
        required=False,
        default=None,
        format="%.3f",
        min_value=None,
        max_value=None,
        step=None
    )
}
st.dataframe(data=df, width=None, height=None, use_container_width=False, hide_index=True, column_order=None, column_config=column_config)




# ###
# dfX = pd.DataFrame(columns=['Student', 'Class', 'Title', 'Content'])

# st.data_editor(
#     dfX,
#     width=None,
#     height=None,
#     use_container_width=True,
#     hide_index=False,
#     column_order=None,
#     column_config=None,
#     num_rows="dynamic",
#     disabled=False,
#     key=None,
#     on_change=None,
#     args=None,
#     kwargs=None
# )
