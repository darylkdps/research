import streamlit as st
import numpy as np
import pandas as pd
import xlsxwriter
import re
from io import BytesIO
from textblob import TextBlob

st.set_page_config(
    page_title='IN-Learning Sentiment Analysis Application',
    page_icon='📚',
    layout='wide',
    initial_sidebar_state='collapsed'
    )

caption_placeholder = st.empty()
st.title('IN-Learning Sentiment Analysis Application')


def sent_polarity_range(sent_num: float) -> str:
    match sent_num:
        case sent_num if -1 <= sent_num < -0.05:
            return "Negative"
        case sent_num if -0.05 <= sent_num <= 0.05:
            return "Neutral"
        case sent_num if 0.05 < sent_num <= 1:
            return "Positive"

def sent_subjectivity_range(sent_num: float) -> str:
    # 0.0 is very objective and 1.0 is very subjective.
    match sent_num:
        case sent_num if 0 <= sent_num < 0.5:
            return "Objective"
        case sent_num if 0.5 <= sent_num <= 1:
            return "Subjective"

uploaded_file = st.file_uploader(
    label='''Load a csv document with 2 columns for sentiment analysis.\n\nData schema: First column (*index*), second column (*text to analyse*)''',
    type=['csv'],
    accept_multiple_files=False,
    disabled=False,
    )

if uploaded_file is not None:
    placeholder = st.empty()
    df = pd.read_csv(uploaded_file, header=0, index_col=None, engine='pyarrow')

    try:
        with st.spinner('Procesing ...'):
            assert len(df.columns) == 2

            col1_name = df.columns[0]
            col2_name = df.columns[1]

            df[col2_name + '_sent_polarity_num']     = df[col2_name].apply(lambda row: np.round(TextBlob(row).polarity, 2))
            df[col2_name + '_sent_subjectivity_num'] = df[col2_name].apply(lambda row: np.round(TextBlob(row).subjectivity, 2))
            df[col2_name + '_sent_polarity_cat']     = df[col2_name + '_sent_polarity_num'].apply(lambda row: sent_polarity_range(row))
            df[col2_name + '_sent_subjectivity_cat'] = df[col2_name + '_sent_subjectivity_num'].apply(lambda row: sent_subjectivity_range(row))
            csv_doc = df.to_csv(index=False, encoding='utf-8') #.encode('utf-8')

            st.markdown('Results:')

            st.data_editor(
                data=df,
                width=None,
                height=None,
                use_container_width=True,
                hide_index=True,
                column_order=None,
                # column_config=df_word_pairs_column_config,
                disabled=True
                )
            
            st.download_button(
                label='Download results as csv',
                data=csv_doc,
                file_name='sentiment_analysis_results.csv',
                mime='text/csv',
                )
            
            

            st.toast(f'🎉 Sentiment analysis of *{uploaded_file.name}* completed.')


        
    except:
        st.markdown('The file is not in a valid format.')
        
        

    # with placeholder:
    #     st.data_editor(
    #         data=df,
    #         width=None,
    #         height=None,
    #         use_container_width=True,
    #         hide_index=True,
    #         column_order=None,
    #         # column_config=df_word_pairs_column_config,
    #         disabled=True
    #         )
    
    # compute_sentiment_button = st.button(
    #     'Compute sentiment',
    #     key='compute_sentiment_button',
    #     help='Compute sentiment polarity and subjectivity.',
    #     on_click=None,
    #     type='secondary',
    #     disabled=False,
    #     use_container_width=False
    #     )


    # if compute_sentiment_button:
        






# @st.cache_resource(ttl=3600)
# def load_model():
#     return spacy.load('en_core_web_lg')
# nlp = load_model()

# @st.cache_resource(ttl=3600)
# def clean_text(text: str) -> str:
#     '''Clean text by remapping specified characters, then removing extraneous spaces.'''
#     # Characters mapping table
#     dict_map = {
#         '“':  '"',  # LEFT DOUBLE QUOTATION MARK: QUOTATION MARK
#         '”':  '"',  # RIGHT DOUBLE QUOTATION MARK: QUOTATION MARK
#         '‘':  "'",  # LEFT SINGLE QUOTATION MARK: APOSTROPHE
#         '’':  "'",  # RIGHT SINGLE QUOTATION MARK: APOSTROPHE
#         '–':  '-',  # EN DASH: HYPHEN-MINUS
#         '\t': ' ',  # Horizontal tab
#         '\n': ' ',  # Line feed
#         '\v': ' ',  # Vertical tab
#         '\f': ' ',  # Form feed
#         '\r': ' ',  # Carriage return
#     }

#     # Map chacters
#     mapped_text = text.translate(str.maketrans(dict_map))

#     # Remove extraneous spaces
#     extraneous_spaces_stripped = re.compile(' {1,}').sub(' ', mapped_text)

#     # Remove leading and trailing whitespaces
#     extraneous_spaces_stripped = extraneous_spaces_stripped.strip()

#     return extraneous_spaces_stripped

# @st.cache_resource
# def tokenise(text: str) -> list:
#     '''Convert string to spacy.Language object and get valid and lemmatised tokens.
#     Returns a pd.Series:
#         pd.Series(
#             [
#                 [
#                     ('A1', 'B1', 'C1', 'D1', 0.001),
#                     ('A2', 'B2', 'C2', 'D2', 0.002)
#                 ],
#                 [
#                     ('E1', 'F1', 'G1', 'G1', 0.002),
#                     ('E2', 'F2', 'G2', 'G2', 0.002)
#                 ]
#             ]
#         )
#     '''
#     open_class_pos = ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']  # Excluded 'INTJ'

#     doc = nlp(text)
#     tokens = [
#         token.lemma_
#         for token in doc
#         if (
#             token.is_alpha and
#             token.ent_type_ == '' and
#             token.pos_ in open_class_pos and
#             not token.is_stop and
#             token.has_vector and
#             len(token) > 1
#         )
#     ]
#     tokens = list(map(str.lower, tokens))

#     return tokens

# # @st.cache_resource
# def get_word_pairs(row):
#     row_index1 = row['Index 1']
#     row_index2 = row['Index 2']
#     row_title = row['Title']
#     row_content = row['Content']

#     title_tokenised = tokenise(clean_text(row_title))
#     content_tokenised = tokenise(clean_text(row_content))

#     data = [
#         (
#             row_index1,
#             row_index2,
#             title_token,
#             content_token,
#             nlp.vocab[title_token].similarity(nlp.vocab[content_token])
#         )
#         for title_token in title_tokenised for content_token in content_tokenised
#     ]

#     return data

# def export_word_pairs():
#     '''documentation here'''
#     df_cs = sl_de.apply(get_word_pairs, axis=1)
#     df_cs_flat = [tup for lst in df_cs.to_list() for tup in lst]

#     df_word_pairs = pd.DataFrame(
#         data=df_cs_flat,
#         columns=['Index 1', 'Index 2', 'Word Pair (Title)', 'Word Pair (Content)', 'Cosine Similarity']
#         )
    
#     df_word_pairs_column_config = {
#         'Index 1': st.column_config.TextColumn(
#             label=None,
#             width=None,
#             help='First unique identifier',
#             disabled=True,
#             required=False,
#             default=None,
#             max_chars=None,
#             validate=None
#         ),
#         'Index 2': st.column_config.TextColumn(
#             label=None,
#             width=None,
#             help='Second unique identifier',
#             disabled=True,
#             required=False,
#             default=None,
#             max_chars=None,
#             validate=None
#         ),
#         'Word Pair (Title)': st.column_config.TextColumn(
#             label=None,
#             width=None,
#             help='Title component of word pair',
#             disabled=True,
#             required=False,
#             default=None,
#             max_chars=None,
#             validate=None
#         ),
#         'Word Pair (Content)': st.column_config.TextColumn(
#             label=None,
#             width=None,
#             help='Content component of word pair',
#             disabled=True,
#             required=False,
#             default=None,
#             max_chars=None,
#             validate=None
#         ),
#         'Cosine Similarity': st.column_config.NumberColumn(
#             label=None,
#             width=None,
#             help='Semantic distance between title and content words. Range: -1.0 to 1.0.',
#             disabled=True,
#             required=False,
#             default=None,
#             format='%.3f',
#             min_value=None,
#             max_value=None,
#             step=None
#             )
#         }
                
#     st.dataframe(
#         data=df_word_pairs,
#         width=None,
#         height=None,
#         use_container_width=True,
#         hide_index=True,
#         column_order=None,
#         column_config=df_word_pairs_column_config
#         )
    
#     buffer = BytesIO()
#     with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
#         # df_word_pairs.to_excel(writer, sheet_name='Word Pairs', index=False)
#         # writer.close()

#         df_word_pairs.to_excel(writer, sheet_name='Word Pairs', startrow=1, header=False, index=False, float_format="%.3f")
#         worksheet = writer.sheets['Word Pairs']

#         # Get the dimensions of the dataframe.
#         (max_row, max_col) = df_word_pairs.shape

#         # Create a list of column headers, to use in add_table().
#         column_settings = [{'header': column} for column in df_word_pairs.columns]

#         # Add the Excel table structure. Pandas will add the data.
#         worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings, 'name': 'tbl_Word_Pairs'})

#         # Make the columns wider for clarity.
#         worksheet.set_column(0, max_col - 1, 15)

#         # Close the Pandas Excel writer and output the Excel file.
#         # writer.close()
#         #####

#     st.download_button(
#         label='Download data as .xlsx',
#         data=buffer,
#         file_name='Word Pairs.xlsx',
#         mime='application/vnd.ms-excel'
#         )

# if 'sl_data_editor_key' not in st.session_state:
#     st.session_state['sl_data_editor_key'] = 0

# def reset_sl_data_editor_key():
#     if len(sl_de) != 0:
#         st.session_state['sl_data_editor_key'] += 1

# metadata1 = {key:value for key, value in nlp.meta.items() if key in {'lang', 'name', 'version', 'vectors'}}
# metadata2 = f"*spaCy={spacy.__version__}, {metadata1['lang']}_{metadata1['name']}={metadata1['version']} ({metadata1['vectors']['vectors']} vectors with {metadata1['vectors']['width']} dimensions)*"
# caption_placeholder.caption(metadata2)

# df_empty = pd.DataFrame(columns=['Index 1', 'Index 2', 'Title', 'Content'])
# df_empty_dtypes = {'Index 1': 'string[pyarrow]', 'Index 2': 'string[pyarrow]', 'Title': 'string[pyarrow]', 'Content': 'string[pyarrow]'}
# df_empty = df_empty.astype(df_empty_dtypes)

# sl_data_editor_column_config = {    
#     'Index 1': st.column_config.TextColumn(
#         label=None,
#         width=None,
#         help='First unique identifier',
#         disabled=False,
#         required=True,
#         default=None,
#         max_chars=None,
#         validate=None
#     ),
#     'Index 2': st.column_config.TextColumn(
#         label=None,
#         width=None,
#         help='Second unique identifier',
#         disabled=False,
#         required=None,
#         default=None,
#         max_chars=None,
#         validate=None
#     ),
#     'Title': st.column_config.TextColumn(
#         label=None,
#         width=None,
#         help='Topic',
#         disabled=False,
#         required=True,
#         default=None,
#         max_chars=None,
#         validate=None
#     ),
#     'Content': st.column_config.TextColumn(
#         label=None,
#         width=None,
#         help='Elaboration of topic',
#         disabled=False,
#         required=True,
#         default=None,
#         max_chars=None,
#         validate=None
#     )
# }

# sl_de = st.data_editor(
#     df_empty,
#     width=None,
#     height=None,
#     use_container_width=True,
#     hide_index=True,
#     column_order=None,
#     column_config=sl_data_editor_column_config,
#     num_rows='dynamic',
#     disabled=False,
#     key=f"data_editor_{st.session_state['sl_data_editor_key']}",
#     on_change=None,
#     args=None,
#     kwargs=None
# )

# col1, col2, _ = st.columns([0.2, 0.2, 0.6], gap='small')

# with col1:
#     clear_data_editor_button = st.button(
#         'Clear',
#         key='clear_data_editor_button',
#         help='Clear input',
#         on_click=reset_sl_data_editor_key,
#         type='secondary',
#         disabled=False,
#         use_container_width=False
#     )

# with col2:
#     compute_data_editor_button = st.button(
#         'Compute',
#         key='compute_data_editor_button',
#         help='Compute cosine similarities',
#         on_click=None,
#         type='secondary',
#         disabled=False,
#         use_container_width=False
#     )

# if clear_data_editor_button:
#     # st.write(st.session_state['sl_data_editor_key'])
#     st.write(len(sl_de))
#     pass
    

# if compute_data_editor_button:
#     if len(sl_de) != 0:
#         sl_de = sl_de.applymap(clean_text, na_action='ignore')
#         export_word_pairs()