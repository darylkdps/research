import streamlit as st
import numpy as np
import pandas as pd
import unicodedata
import pickle
import spacy

st.set_page_config(
    page_title='NLP Task',
    page_icon='📏',
    layout='wide',
    initial_sidebar_state='collapsed'  # auto, expanded, collapsed
    )

@st.cache_resource
def load_model():
    spacy.require_cpu()
    return spacy.load('en_core_web_lg')
nlp = load_model()

caption_placeholder = st.empty()
st.title('NLP Tasks')
metadata1 = {key:value for key, value in nlp.meta.items() if key in {'lang', 'name', 'version', 'vectors'}}
metadata2 = f"*spaCy={spacy.__version__}, {metadata1['lang']}_{metadata1['name']}={metadata1['version']} ({metadata1['vectors']['vectors']} vectors with {metadata1['vectors']['width']} dimensions)*"
# caption_placeholder.caption(metadata2)

# Tokenisation
st.header('Tokenisation')

tokenisation_input1 = st.text_area(
    label='Input some text:',
    value='''Many words map to one token, but some don't: indivisible.
Unicode characters like emojis may be split into many tokens containing the underlying bytes: 🤚🏾
Sequences of characters commonly found next to each other may be grouped together: 1234567890''',
    placeholder='',
    height=100,
    max_chars=250,
    key='tokenisation_input1',
    )
if st.button('Tokenise', key='Tokenise_button1'):
    doc = nlp(tokenisation_input1)
    st.write([token.text for token in doc])

tokenisation_input2 = st.text_area(
    label='Input some text:',
    value='''In “How We Think: Digital Media and Contemporary Technogenesis”, Hayles (2012) described distant reading as human-assisted computer reading, where humans use computer algorithms to “analyse patterns in large textual corpora where size makes human reading of the entirety impossible” (p. 70).''',
    placeholder='',
    height=100,
    max_chars=250,
    key='tokenisation_input2',
    )
if st.button('Tokenise', key='Tokenise_button2'):
    doc = nlp(tokenisation_input2)
    st.write([token.text for token in doc])

# Lemmatisation
st.header('Lemmatisation')

lemmatisation_input1 = st.text_area(
    label='Input some text:',
    value='''kites babies dogs flying smiling driving died tried feet bank swim swam swum swimming''',
    height=100,
    max_chars=250,
    key='lemmatisation_input1',
    )
if st.button('Lemmatise', key='Lemmatise_button1'):
    doc = nlp(lemmatisation_input1)
    st.write([f"text: '{token.text}', lemma: '{token.lemma_}'" for token in doc])    

lemmatisation_input2 = st.text_area(
    label='Input some text:',
    value='''Using the breaststroke technique, he swam the English Channel.''',
    height=100,
    max_chars=250,
    key='lemmatisation_input2',
    )
if st.button('Lemmatise', key='Lemmatise_button2'):
    doc = nlp(lemmatisation_input2)
    st.write([f"text: '{token.text}', lemma: '{token.lemma_}'" for token in doc])
    st.markdown(spacy.displacy.render(doc, style='dep', options={'bg': 'black', 'color': 'white', 'compact': False}, jupyter=False), unsafe_allow_html=True)

# Stopwords
st.header('Stopwords removal')

stopwords_input1 = st.text_area(
    label='Input some text:',
    value='''The Dean, Academic and Strategic Development, oversees the Office of Strategic Planning & Academic Quality and the Centre for Innovation in Learning.''',
    height=100,
    max_chars=250,
    key='stopwords_input1',
    )
if st.button('Remove stopwords', key='stopwords_button1'):
    doc = nlp(stopwords_input1)
    st.write([token.text for token in doc if not token.is_stop and len(token.text) > 1])
 
# Part of Speech
st.header('Part of speech')

pos_input1 = st.text_area(
    label='Input some text:',
    value='''As an institute within a world-class research university, NIE also offers rigorous graduate education in the form of masters and doctoral programmes for local and international students.''',
    height=100,
    max_chars=250,
    key='pos_input1',
    )
if st.button('See part of speech', key='pos_button1'):
    doc = nlp(pos_input1)
    st.write([f"text: '{token.text}', POS: {token.pos_} ({spacy.explain(token.pos_)})" for token in doc])

# Named Entity Recognition
st.header('Named entity recognition')

ner_input1 = st.text_area(
    label='Input some text:',
    value='''European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices.''',
    height=100,
    max_chars=250,
    key='ner_input1',
    )
if st.button('See named entities', key='ner_button1'):
    doc = nlp(ner_input1)
    st.write([f"text: '{ent.text}', entity: {ent.label_} ({spacy.explain(ent.label_)})" for ent in doc.ents])
    st.markdown(spacy.displacy.render(doc, style='ent', options={'ents': None}, jupyter=False), unsafe_allow_html=True)


# Semantic Similarity
st.header('Semantic similarity')

col1, col2 = st.columns(2, gap='large')
with col1:
    cos_sim_input1 = st.text_input(
    label='Input first word:',
    value='leadership',
    max_chars=20,
    key='cos_sim_input1',
    )
with col2:
    cos_sim_input2 = st.text_input(
    label='Input second word:',
    value='governance',
    max_chars=20,
    key='cos_sim_input2',
    )
if st.button('See similarity', key='cos_sim_button1'):
    similarity = nlp.vocab[cos_sim_input1].similarity(nlp.vocab[cos_sim_input2])
    st.write(round(similarity, 3))

# # Top N Nearest
# st.header('Top 10 nearest words')

# top_N_nearest_input1 = st.text_input(
#     label='Input a word:',
#     value='leadership',
#     max_chars=20,
#     key='top_N_nearest_input1',
#     )
# if st.button('See top 10 nearest', key='top_N_nearest_button1'):
#     def get_words_nearest(locus_word: str, nearest: int = 10):
#         ms = nlp.vocab.vectors.most_similar(
#             np.asarray([nlp.vocab.vectors[nlp.vocab.strings[locus_word]]]),
#             n=nearest
#             )
#         words = [nlp.vocab.strings[w] for w in ms[0][0]]
#         distances = ms[2]
#         return words
    
#     top_N = get_words_nearest(top_N_nearest_input1, nearest=100)
#     top_N_lower = [word.lower() for word in top_N]
#     top_N_lower = np.array(top_N_lower)
#     _, idx = np.unique(top_N_lower, return_index=True)
#     top_N_lower_list = list(top_N_lower[np.sort(idx)][1:11])
#     top_N_lower_list_cos = [f"{word}, {round(nlp.vocab[top_N_nearest_input1].similarity(nlp.vocab[word]), 3)}" for word in top_N_lower_list]

#     st.write(top_N_lower_list_cos)

# Regex
st.header('Regular expression')

st.markdown('''https://regex101.com/r/EET1sq/1''')
# '''\( *(?:[a-zA-ZÀ-ÖØ-öø-ÿĀ-žȀ-ț.& -]{2,}(?: *, *[12]\d{3}[a-z]?(?: *, *p\. *\d+)?)+)(?: *; *[a-zA-ZÀ-ÖØ-öø-ÿĀ-žȀ-ț.& -]{2,}(?: *, *[12]\d{3}[a-z]?(?: *, *p\. *\d+)?)+)* *\)'''
# '''... study (Gašević, 1999, p. 19999) found ...
# ... study (Gašević, 1999a, 1999b) found ...
# ... study (Csikszentmihalyi, 1999, p.23, 2000, p.34) found ...
# ... study (González-Cruz, 1999, 2000) found ...
# ... study ( de Bono, 1999, 2000, 2001) found ...
# ... study (Reiter-Palmon & Webb, 1999 ) found ...
# ... study (Zhang & Webb, 1999 , 2000) found ...'''

# Token Matcher
st.header('Token matching')

st.markdown('''https://demos.explosion.ai/matcher''')

# Unicode Normalisation
st.header('Unicode normalisation')

    # NFD:  Normalization Form Canonical Decomposition
    # NFC:  Normalization Form Canonical Composition
    # NFKD: Normalization Form Compatibility Decomposition
    # NFKC: Normalization Form Compatibility Composition  - The best form to go with for normalisation of text.

    # D: Characters are decomposed by canonical equivalence, and multiple combining characters are arranged in a specific order.
    # C: Characters are decomposed and then recomposed by canonical equivalence.
    # KD: Characters are decomposed by compatibility, and multiple combining characters are arranged in a specific order.
    # KC: Characters are decomposed by compatibility, then recomposed by canonical equivalence.

unicode_normalisation_input1 = st.text_area(
    label='Input some text:',
    value='''Gašević (2000) study's found ... Csíkszentmihályi's work on ... Chulvi, González-Cruz, and Aguilar-Zambrano (2013) found ... Björk et al.'s (2017) research into ...''',
    height=100,
    max_chars=250,
    key='unicode_normalisation_input1',
    )
if st.button('Normalise', key='unicode_normalisation_buttonKD'):
    normalised = unicodedata.normalize('NFKD', unicode_normalisation_input1).encode(encoding='ascii', errors='ignore').decode('ascii')
    st.text(normalised)

# Vectorisation
st.header('Vectorisation')

vector_input1 = st.text_area(
    label='Input some text:',
    value='''The CJ Koh Professorship has been made possible through a generous donation by the late Mr Ong Tiong Tat, executor of the late lawyer Mr Koh Choon Joo’s (CJ Koh) estate, to the Nanyang Technological University Endowment Fund.''',
    height=100,
    max_chars=250,
    key='vector_input1',
    )
if st.button('Vectorise', key='vectorise_button1'):
    st.code(list(nlp(vector_input1).vector))

st.subheader('*Machine learning*')

st.markdown('''Reference true news:

Turkish authorities have issued detention warrants for 216 people, including former finance ministry personnel, suspected 
of having links to last year's failed coup attempt, the state-run Anadolu news agency said on Wednesday. It said 17 former 
finance ministry personnel had been detained so far and another 65 were sought over alleged links to Gulen's network, Anadolu 
said. Separately, authorities carried out operations across 40 provinces targeting private imams believed to be recruiting members 
to the network of U.S.-based cleric Fethullah Gulen from Turkey's armed forces. Ankara blames Gulen for orchestrating the 
July 15 coup attempt last year and has repeatedly demanded the United States extradite him, so far in vain. Gulen denies 
involvement. In the aftermath of the coup, more than 50,000 people have been jailed pending trial and some 150,000 have 
been sacked or suspended from their jobs in the military, public and private sector. The extent of the purges has unnerved 
rights groups and Turkey's Western allies, who fear President Tayyip Erdogan is using the abortive putsch as a pretext 
to stifle dissent. The government, however, says the measures are necessary due to the gravity of the threats it is facing 
following the military coup attempt, in which 240 people were killed.
''')

st.markdown('''Reference fake news:

Hollywood stars took to social media Wednesday to express outrage over President Donald Trump's announcement that transgender 
people will not be allowed to serve in the U.S. military, reversing former President Obama's decision last year allowing 
them to do so. After consultation with my Generals and military experts, please be advised that the United States Government 
will not accept or allow Transgender individuals to serve in any capacity in the U.S. Military, President Trump tweeted 
Wednesday morning. Our military must be focused on decisive and overwhelming victory and cannot be burdened with the tremendous 
medical cost and disruption that transgender in the military would entail. Thank you. The news spurred some stars to viciously 
attack the president, with some calling Trump a cruel bigot. Star Trek actor and gay activist George Takei sent what appeared 
to be a threat to Trump, warning that he  just pissed off the wrong community  and said the president  will regret this 
action. Donald: With your ban on trans people from the military, you are on notice that you just pissed off the wrong community. 
You will regret it. Takei also tweeted: History shall record that you are not only the stupidest, most incompetent president 
ever, but also the cruelest and pettiest.
''')

loaded_minmaxscaler_model = pickle.load(open('pages/minmaxscaler_model.sav', 'rb'))
loaded_kneighborsclassifier_model = pickle.load(open('pages/kneighborsclassifier_model.sav', 'rb'))

true_news_input1 = st.text_area(
    label='Input some news:',
    value='''An old review of an academic monograph on agrarian revolutionaries in 1930s China is hardly a political third rail in Beijing today, even by the increasingly sensitive standards of the ruling Communist Party. That such a piece appeared on a list of some 300 scholarly works that Cambridge University Press (CUP) said last week the Chinese government had asked it to block from its website offers clues about the inner workings of China s vast and secretive censorship apparatus, say experts. President Xi Jinping has stepped up censorship and tightened controls on the internet and various aspects of civil society, as well as reasserting Communist Party authority over academia and other institutions, since coming to power in 2012.''',
    height=100,
    max_chars=1000,
    key='true_news_input1',
    )
if st.button('Classify news', key='classify_true_news_button1'):
    doc = nlp(true_news_input1)
    doc_vec = doc.vector.reshape((1, 300))
    doc_vec_scaled = loaded_minmaxscaler_model.transform(doc_vec)
    result = loaded_kneighborsclassifier_model.predict(doc_vec_scaled)
    result_msg = ':green[This news is likely true.]' if result else ':red[This news is likely fake.]'
    st.markdown(result_msg)


fake_news_input1 = st.text_area(
    label='Input some news:',
    value='''Our reality is carefully constructed by powerful corporate, political and special interest sources in order to covertly sway public opinion. Blatant lies are often televised regarding terrorism, food, war, health, etc. They are fashioned to sway public opinion and condition viewers to accept what have become destructive societal norms. The practice of manipulating and controlling public opinion with distorted media messages has become so common that there is a whole industry formed around this.''',
    height=100,
    max_chars=1000,
    key='fake_news_input1',
    )
if st.button('Classify news', key='classify_fake_news_button1'):
    doc = nlp(fake_news_input1)
    doc_vec = doc.vector.reshape((1, 300))
    doc_vec_scaled = loaded_minmaxscaler_model.transform(doc_vec)
    result = loaded_kneighborsclassifier_model.predict(doc_vec_scaled)
    result_msg = ':green[This news is likely true.]' if result else ':red[This news is likely fake.]'
    st.markdown(result_msg)