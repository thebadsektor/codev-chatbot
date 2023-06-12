import os

from langchain.llms import OpenAI, LLMChain
from langchain.tools import DuckDuckGoSearchRun 
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, create_csv_agent

from streamlit_chat import message
import streamlit as st

from typing import List, Union
import pandas as pd

# api_key = os.environ.get("OPENAI_API_KEY")
api_key = os.environ.get(st.secrets["OPENAI_API_KEY"])

# API key is set as an environment variable
df = pd.read_csv('data/knowledge-base.csv')
# print(df[0:10])

csv_path = 'data/knowledge-base.csv'

# Define which tools the agent can use to answer user queries
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]

# Set up the base template
template = """Answer the following questions as best you can, but speaking as compasionate medical professional. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to answer as a compansionate medical professional when giving your final answer.

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
    
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

## Custom Output Parser
# The output parser is responsible for parsing the LLM output into AgentAction and AgentFinish. 
# This usually depends heavily on the prompt used.
# This is where you can change the parsing to do retries, handle whitespace, etc
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

# Initialise session state variables.
# Session State is a way to share variables between reruns, for each user session. 
# In addition to the ability to store and persist state.
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

csv_agent = create_csv_agent(OpenAI(temperature=0), csv_path, verbose=True)
llm = OpenAI(temperature=0)
# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent_executor = AgentExecutor.from_agent_and_tools(agent=csv_agent, 
                                                    tools=tools, 
                                                    verbose=True)

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
st.set_page_config(page_title="", page_icon=":robot_face:")
st.markdown(
    "<h1 style='text-align: center;'>CoDev Chatbot</h1>"
    + "<p style='text-align: center;'>Role: University Assistant</p>", 
    unsafe_allow_html=True)

st.sidebar.title("Welcome")
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Gerald Villaran")
q_button = st.sidebar.button("Test", key="test")
ddg_button = st.sidebar.button("DuckDuckGo", key="duckduckgo")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = csv_agent.run("Look for similar questions in the intent column and find the answer from the response column. Here is the question:" + user_input)
        # output = csv_agent.run("Look for similar questions in the intent column and find the answer from the response column. Here is the question:" + user_input + ". If the question is not related to 'Enrollment' just say 'Sorry.' ")
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if q_button:
    p = "Categorize the whole dataframe into topics. put total number of questions per category. use practical vocabulary. use bulleted list."
    output = csv_agent.run(p)
    st.session_state['past'].append(p)
    st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

