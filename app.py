import os
from langchain.llms import OpenAI
import pandas as pd
from langchain.agents import create_csv_agent
import streamlit as st

# api_key = os.environ.get("OPENAI_API_KEY")
api_key = os.environ.get(st.secrets["OPENAI_API_KEY"])

if api_key:
    print("OpenAI API key is good.")
else:
    # API key is not set
    print("OpenAI API key is not provided as an environment variable.")

# API key is set as an environment variable
df = pd.read_csv('data/knowledge-base.csv')
print(df[0:10])

csv_path = 'data/knowledge-base.csv'

csv_agent = create_csv_agent(OpenAI(temperature=0), csv_path, verbose=True)
output = csv_agent.run("Look for similar questions in the intent column and find the answer from the response column. Here is the question: What is LSPU?")
print(output)
