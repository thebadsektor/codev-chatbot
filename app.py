import os
from langchain.llms import OpenAI
import pandas as pd
from langchain.agents import create_csv_agent
import streamlit as st
from streamlit_chat import message

# api_key = os.environ.get("OPENAI_API_KEY")
api_key = os.environ.get(st.secrets["OPENAI_API_KEY"])

if api_key:
    print("OpenAI API key is good.")
else:
    # API key is not set
    print("OpenAI API key is not provided as an environment variable.")

# API key is set as an environment variable
df = pd.read_csv('data/knowledge-base.csv')
# print(df[0:10])

csv_path = 'data/knowledge-base.csv'

csv_agent = create_csv_agent(OpenAI(temperature=0), csv_path, verbose=True)
question = "Look for similar questions in the intent column and find the answer from the response column. Here is the question: What is LSPU?"

def respond():
    output = csv_agent.run(question)
    print(output)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Setting page title and header
st.set_page_config(page_title="AVA", page_icon=":robot_face:")
st.markdown(
    "<h1 style='text-align: center;'>CoDev Chatbot</h1>"
    + "<p style='text-align: center;'>Role: University Assistant</p>", 
    unsafe_allow_html=True)

st.sidebar.title("Welcome")
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Gerald Villaran<br>Chatbot")
q_button = st.sidebar.button("Test", key="test")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

        if submit_button:
            respond()
