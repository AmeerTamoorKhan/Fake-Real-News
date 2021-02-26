import streamlit as st
import tensorflow.keras as keras
import pickle
import numpy as np
import pandas as pd

example_data = [('Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing', 'Fake'),
                ('U.S. military to accept transgender recruits on Monday: Pentagon', 'Real'),
                ('New York governor questions the constitutionality of federal tax overhaul', 'Real'),
                ('Trump Just Made A Major Threat To The Press, Jake Tapper’s Response Is Perfect (VIDEO)', 'Fake'),
                ('Cambodia revokes diplomatic passports of banned opposition members', 'Real')]

df = pd.DataFrame(example_data, columns=['News', 'Status'])


def default():
    st.header('Working Demonstration')
    st.video('imgs/FakeRealNews.mp4')
    st.header('How It Works')
    st.markdown('''
        <p>The Fake News Detector is a research project to classify the FAKE news and REAL news. Deep learning model is
        trained over 40,000 news taken from a benchmark dataset. The dataset composed of political news mostly, so may
        not be compatible with news from other walks of life. The model has the accuracy of over 90% with validation 
        and test dataset.</p>
        <h4><strong>#python</strong> <strong>#deeplearning</strong> <strong>#pandas</strong> <strong>#numpy</strong>
        <strong>#streamlit</strong>
    ''', unsafe_allow_html=True)

def model(test_news):
    X = np.zeros((1, 42))
    model = keras.models.load_model('RealFakeNewsBest')
    #model.summary()
    with open('t_token.pickle', 'rb') as f:
        token = pickle.load(f)

    text = np.array([test_news])
    text = token.texts_to_sequences(text)

    ind = len(text[0])
    X[0, 0:ind] = text[0]

    result = np.round(model.predict(X))

    return result


st.set_page_config('News Analyzer')
st.sidebar.title('Welcome')
page = st.sidebar.radio(' ', options=['Home', 'Fake News Detector'])
st.sidebar.markdown('''<h3>Created By: Ameer Tamoor Khan</h3>
                    <h4>Github : <a href="https://github.com/AmeerTamoorKhan" target="_blank">Click Here </a></h4> 
                    <h4>Email: drop-in@atkhan.info</h4> ''', unsafe_allow_html=True)

if page == 'Home':
    default()
elif page == 'Fake News Detector':
    cols = st.beta_columns((0.15, 0.85))
    cols[0].image('imgs/Blank.png', width=100)
    cols[1].title("Fake News Detector")
    st.subheader("Please Enter The News")
    test_news = st.text_input(" ")
    enter = st.button("Enter")

    result = model(test_news)
    cols = st.beta_columns(3)
    if enter:
        if int(result[0][0]) == 0:
            cols[1].image('imgs/Fake_News.png', width=300)
        elif int(result[0][0]) == 1:
            cols[1].image('imgs/Real_News.png', width=300)
    else:
        cols[1].image('imgs/Blank.png', width=300)

    st.subheader("Examples")
    st.table(df)





