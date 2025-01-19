import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_experimental.tools import PythonREPLTool
from langchain.prompts import PromptTemplate
from requests.auth import HTTPBasicAuth
import requests
import logging
import os

# Load environment variables (e.g., OpenAI API key)
from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define tools
python_tool = PythonREPLTool()

python_repl = Tool(
    name="Python REPL",
    func=python_tool.run,
    description="Use this tool to execute Python code. Useful for calculations or programming-related tasks."
)

tools = [
    python_repl
]

# Define a prompt template (optional customization)
prompt_template = PromptTemplate(
    template="You are a helpful AI assistant. Answer the user's question or execute their request using the available tools: {tools}.",
    input_variables=["tools"]
)

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    prompt=prompt_template
)

def chat_with_agent(user_input):
    try:
        response = agent.run(user_input)
        return response
    except Exception as e:
        return f"Error: {e}"

# Gradio Interface
iface = gr.Interface(
    fn=chat_with_agent,
    inputs="text",
    outputs="text",
    title="LangChain AI Agent",
    description="Chat with an AI agent powered by LangChain and GPT-4."
)

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860))
    )