import os

from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun 
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, create_csv_agent

from streamlit_chat import message
import streamlit as st

from typing import List, Union
import pandas as pd
import re

# api_key = os.environ.get("OPENAI_API_KEY")
api_key = os.environ.get(st.secrets["OPENAI_API_KEY"])

# API key is set as an environment variable
df = pd.read_csv('data/knowledge-base.csv')
# print(df[0:10])

csv_path = 'data/knowledge-base.csv'

agent = create_csv_agent(OpenAI(temperature=0), csv_path, verbose=True)
print(agent.agent.llm_chain.prompt.template)
# output = agent.run("how many rows in the dataframe?")