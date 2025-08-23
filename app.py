from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import time
import streamlit as st
import warnings
import matplotlib.pyplot as plt
import json
from streamlit_lottie import st_lottie
import speech_recognition as sr
import toml

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
def load_lottiefile(filepath:str):
    with open(filepath,'r')as f:
        return json.load(f)
lottie_coding=load_lottiefile('animation.json')
# st.title('Sentiment Analysis',divider ='rainbow')
def load_lottiefile_voice(filepath:str):
    with open(filepath,'r',encoding='utf-8')as f:
        return json.load(f)
lottie_anime_voice=load_lottiefile_voice('Animation - speech.json')

@st.cache_data()
def mode():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    time.sleep(10)
    return tokenizer ,model , MODEL
tokenizer,model,MODEL=mode()

# st.header('Sentiment Analyzer App', divider='rainbow')
st.title('Sentiment Analyzer App')
# sentence = st.text_input('Enter your sentence ')
prompt = st.chat_input("Say something")  
# messages = st.container(height=300)
if st.button("Voice", type="primary"):
    st_lottie(lottie_anime_voice,speed=1,reverse=False,loop=False,quality='low',height=None)
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        print("Listening...")
        audio_data = recognizer.listen(source)  # Listen for audio

        try:
            # Recognize speech using Google's speech recognition
            text = recognizer.recognize_google(audio_data)
            st.title(text)
            prompt = text
        except sr.UnknownValueError:
            print(" Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Speech Recognition service; {e}")
            
if prompt!=None:
    
    
   # Run for Roberta Model
    encoded_text = tokenizer(prompt, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'neg' : scores[0],
        'neu' : scores[1],
        'pos' : scores[2]
    }
    print(scores_dict)
    
    max_key = max(scores_dict, key=scores_dict.get)
    
    if max_key=='neu':
        guage_value=1.60
    elif max_key=='pos':
        guage_value=0.5
    else:
        guage_value=2.7
    
    # max_value = scores_dict[max_key]
    
    # max_value=float(max_value)
    
    # result_score_value = round(max_value,2)
    
    # pos:0.5,neu:1.60,neg:2.7
    
    # colors = ['#4dab6d', "#72c66e", "#c1da64", "#f6ee54", "#fabd57", "#f36d54", "#ee4d55"]
    colors = ['#4dab6d', "#4dab6d", "#fabd57", "#fabd57", "#fabd57", "#ee4d55", "#ee4d55"]
    values = [100,90,75,60,45,30,15, 0]
    
    x_axis_vals = [0, 0.44, 0.88,1.32,1.76,2.2,2.64]
    
    fig = plt.figure(figsize=(18,18))
    
    ax = fig.add_subplot(projection="polar");
    
    ax.bar(x=[0, 0.44, 0.88,1.32,1.76,2.2,2.64], width=0.5, height=0.5, bottom=2,
           linewidth=3, edgecolor="white",
           color=colors, align="edge");
  
    plt.annotate("Positive", xy=(0.16,2.1), rotation=-75, color="white", fontweight="bold");
    plt.annotate("Positive", xy=(0.65,2.08), rotation=-55, color="white", fontweight="bold");
    plt.annotate("Neutral", xy=(1.14,2.1), rotation=-32, color="white", fontweight="bold");
    plt.annotate("Neutral", xy=(1.62,2.2), color="white", fontweight="bold");
    plt.annotate("Neutral", xy=(2.08,2.25), rotation=20, color="white", fontweight="bold");
    plt.annotate("Negative", xy=(2.46,2.25), rotation=45, color="white", fontweight="bold");
    plt.annotate("Negative", xy=(3.0,2.25), rotation=75, color="white", fontweight="bold");
    
    for loc, val in zip([0, 0.44, 0.88,1.32,1.76,2.2,2.64, 3.14], values):
        plt.annotate(val, xy=(loc, 2.5), ha="right" if val<=20 else "left");
    
    plt.annotate("  ", xytext=(0,0), xy=(guage_value, 1.7),
                 arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="black", shrinkA=0),
                 bbox=dict(boxstyle="circle", facecolor="black", linewidth=2.0, ),
                 fontsize=45, color="white", ha="center"
                );
    
    
    plt.title("SENTIMENT CHART", loc="center", pad=20, fontsize=35, fontweight="bold");
    
    ax.set_axis_off();
    st.pyplot(plt)
elif prompt==None:
    st_lottie(lottie_coding,speed=1,reverse=False,loop=True,quality='low',height=None)
else:
    st_lottie(lottie_coding,speed=1,reverse=False,loop=True,quality='low',height=None)
