import streamlit as st

st.set_page_config(
    page_title='Ideation model',
    page_icon='💡',
    layout='wide',
    initial_sidebar_state='auto'
    )

st.title('Reflections on appraising ideation as a future-ready habit')

st.markdown('---')
st.header('Project objective')
st.markdown(
    '''
    Appraise the ideation quality of university students' essays computationally.
    '''
    )

st.markdown('---')
st.header('Hardware resource')
st.markdown(
    '''
    Laptop\n
    \tCPU:\tIntel i9-11950H @ 2.60Hz
    \tGPU:\tNVIDIA RTX A3000 with 6GB VRAM
    \tRAM:\t32 GB
    '''
    )

st.markdown('---')
st.header('Data resource')
st.markdown(
    '''
    5 essays from a doctoral program on educational leadership at inception.

    23 essays from a master's program on educational leadership, acquired 5+ months later.
    '''
    )

st.markdown('---')
st.header('Constraint')
st.markdown(
    '''
    Only 5 essays, instead of 500 000, available for exploration at inception. This is not enough to generate word 
    embeddings.
    
    Word embeddings are vector representations of words. E.g., cat = [1, 0, 1, 0, 1], dog = [0, 1, 1, 0, 0]. Each 
    dimension in these vectors represents an aspect of its word's meaning as part of a corpus. The higher the 
    dimension, the better it is at capturing the subtleties of a word's meaning.

    With only 5 essays, moreover, all from the same field, work cannot proceed further because new essays will come 
    with a different set of words. This means new essays cannot be analysed.
    
    A one-man data science team in a university's administrative department does not have the resources to acquire 
    the amount of data required. To proceed further, alternative word embeddings must be sought.

    A decision was made to use spaCy's word-level pretrained word embeddings *en_core_web_lg (version 3.2.0)*. This 
    provides 684 830 unique vectors with 300 dimensions each. In plain English, this means it would be able to 
    recognise words from a wider variety of disciplines.

    However, *en_core_web_lg* is also constantly being updated. It is presently at version 3.5.0 as of this 
    writing. This version provides 514 157 unique vectors with 300 dimensions each. This means different versions 
    will result in different analytical outputs. This also means any measurements based on these word embeddings 
    should be relative rather than absolute. Therefore, an ideation quality of 123 in *en_core_web_lg* version 
    3.2.0 will not be the same as an ideation quality of 123 in *en_core_web_lg* version 3.7.8.

    To address this relative nature, a decision was made to incorporate expert-in-the-loop as part of the appraisal 
    process. It will still stay true to the objective, except that experts will have to be consulted in order for 
    ideation parameters to be determined. This would also be more coherent with creativity as a system.
    '''
    )

st.markdown('---')
st.header('Directions')
st.markdown(
    '''
    The initial direction was to extract the top 10 words in the content proper of an essay most semantically 
    related to its title. And then extract again the top 10 words in the content proper most semantically related 
    to the previous top 10 words.

    This method has its limitations. It highlights semantically related words in individual essays but it does 
    not specifically address the objective. Moreover, it assumes every word ranked in the top 10 is a consequence of 
    ideation. This method has no theoretical coherence with the literature.

    Appraising ideation quality as an engineering endeavour is significantly different from appraising ideation 
    quality as a research endeavour. As a research endeavour, the method and its operational decisions must be 
    based on or be coherent with the literature.

    A literature review was subsequently conducted to further explore and strengthen the ideas and concepts 
    that are cognate.

    As a *future-ready* habit, ideation refers to the production of ideas stemming from divergent as well as 
    convergent thinking, and is closely associated with creativity and innovation. Conceptually, ideation means 
    coming up with ideas. However, ideation is not a standalone concept. It is an intricate part of the bigger 
    process of creative problem solving (CPS). As such, it cannot be reified decontextualised from CPS. In this 
    context, ideation can mean two things. During the problematising phase, ideation can be directed at clarifying or 
    framing problems. During the solutioning phase, ideation can be directed at seeding or refining solutions. The 
    second meaning is adopted here as it is more coherent with our intent.
    
    In the context of CPS, creative ideas are about possibilities not probabilities. Ideas are therefore considered 
    creative not just because they are avant-garde but also because they are practical.
    
    Ideas are not immutable. They evolve. During early stage ideation, emphasis is placed on generating blue-sky 
    ideas. During mid and late stage ideation, emphasis is placed on refining and developing the pool of blue-sky 
    ideas - directed at innovation.

    The quality of an idea can generally be described in terms of of *novelty* (how unique), *variety* (how 
    different), *quality* (how good), and *quantity* (how many). However, novelty, variety, and quality are 
    judgement calls made in relation to a complex system of interacting personal, social, and cultural factors. 
    Outside of this system, creativity does not exist. Unfortunately, technology is not a personality. Therefore, the 
    computational appraisal of ideation quality falls outside of this system. This leaves quantity as the only means 
    in which we can describe the quality of an idea.

    The review of the literature also explored how to analyse large volumes of text. We were specifically inspired by 
    the literature on distant reading. It is an analytical approach described in the digital humanities as the 
    computational study of text. It can be understood as *human-assisted* computer reading.
    
    Following the literature review, operational decisions could be made with regards to key challenges such as:

    1.  What constitute an idea in the context of university students' essays.
    2.  What constitute a word. In linguistics, the concept of a word does not exists.
    3.  What constitute novelty, variety, quality, quantity.
    4.  How to appraise based on novelty, variety, quality, quantity as metrics.
    5.  What constitute a divergent idea. Creativity and CPS have never been about divergent thinking alone.
    6.  What constitute a convergent idea.
    '''
)

st.markdown('---')
st.header('Data preprocessing')
st.markdown(
    '''
    The preprocessing of essays was more problematic than we anticipated.

    Poor digital literacy is a prevalent problem around the world. Knowing how to operate a technology and knowing 
    how to *use* a technology is not the same thing. The master's and doctoral students of the essays were not an 
    exception. Not knowing how to use Word is problematic because it would be impossible to anticipate the myriad of 
    idiosyncratic practices. We expect to find the relevant data in the main document story. The use of idiosyncratic 
    practices, such as writing an entire essay in a table cell, would mean there would be no data to extract. The use 
    of desktop publishing practices to alter the content of an essay to make it look "correct", as opposed to using 
    the appropriate word processing tool to alter how the essay is presented, would also affect the data.

    Poor scholarship was a problem we did not expect to encounter with postgraduate students. Postgraduates are 
    generally expected to know how to reference academic articles. They have had at least 3 years of experience doing 
    so as an undergraduate. Being in the social sciences, they also should have been aware that the use of APA style 
    is standard. We instead encountered essays that used unorthodox or idiosyncractic styles, or a commingling of 
    styles. It also appeared the students did not know how to use referencing tools such Endnote, Zotero, and 
    Mendeley. While a paid software, Endnote is typically made available to students for free at most universities. 
    Even though Endnote has never provided actual support for APA style, it is still sufficient, and more productive, 
    to use the APA style template it provides to manage references. Alternatives like Zotero and Mendeley are free, 
    and not at all inferior to Endnote. Academic references are not ideation per se. We need to reliably determine 
    references so they can be removed.

    The quantity of word pairs can increase ideation score. However, this does not appear to be correlated following 
    preliminary analysis. While images in essays will be ignored, the captions of images will be treated as content 
    proper - as will other irrelevant content. This can increase the ideation score of an essay. One particular essay 
    consisted of 61 pages. Content proper, it consisted of 29 pages with 5364 words excluding references. Its 
    appendix alone consisted of 32 pages with 1972 words. All 28 essays were cleaned manually so this particular 
    essay was intercepted. If there are hundreds of essays, it may not be possible for these essays to be cleaned 
    manually.

    Essays from InsPIRE's ICC courses were also considered. While the data can be ingested, it is also unstructured 
    so we cannot determine where the essay proper begins. It could be in paragraph 3 in one essay and paragraph 15 
    in the next. These essays also do not have a title proper.
    '''
)

st.markdown('---')
st.header('Analytical method')
st.markdown(
    '''
    The analysis is fundamentally based on word pairs as ideas. The words in the essay title are treated as one set 
    of words, and the words in the essay content are treated as another. The word pairs of an essay is the Cartesian 
    product of the two sets of words.

    This approach is *naive*. It assumes every word in the content proper is independent of one another. This is 
    obviously not true. For example, there is only a limited number of adjectives that can be used to qualify a noun. 
    Even so, this approach is not invalid.

    With a word pair, the semantic similarity between its title word and its content word can be quantified using 
    the cosine similarity of its vectors. This vector is dependent on the word embeddings used.

    Cosine similarity, however, is not a coherent unit of analysis. We are specifically interested in how divergent 
    or convergent an idea is. To achieve this, a domain expert needs to be consulted to determine the parameters for 
    divergentness and convergentness, so an idea can be categorised accordingly. We developed a "semantic ruler" to 
    facilitate this process.

    Creativity and CPS is never about divergent thinking alone. Divergent thinking is merely the precursor to 
    convergent thinking. As such, divergent and convergent ideas should also be weighed differently. It is coherent 
    with the literature to weigh convergent ideas higher in this case as it is the immediate prelude to innovation. 
    Ideas, no matter how creative, are useless in the context of CPS if they are not practical.

    Following preliminary analysis, we observed a fairly symmetrical distribution of the cosine similarity of the 
    word pairs. The mean and median in both the master's and doctoral essays are also very close. This was 
    unexpected. Until there are more data, there is not much we can interpret from this finding.
    '''
)

st.markdown('---')
st.header('Future direction')
st.markdown(
    '''
    The narrative needs to be more consistent. Presently, our narrative vacillates between ideation as just another 
    term to describe the summative evaluation of academic performance, and ideation as a future-ready habit distinct 
    from knowledge and skills. There is more value in presenting this work as one that fills a niche area. An area 
    specifically addressing the ongoing development and evaluation of ideation as a habit.

    There was insufficient data to work further. With more data, we could possibly explore examining only open class 
    words such as nouns, verbs, adjectives, and adverbs for further de-noising. We could also explore coreference 
    resolution to translate pronouns to the entity they reference. E.g., The *music* was so loud that *it* could not 
    be enjoyed.
    '''
)