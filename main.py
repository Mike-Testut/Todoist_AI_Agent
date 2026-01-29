from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from todoist_api_python.api import TodoistAPI

load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

todoist = TodoistAPI(todoist_api_key)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                             google_api_key=gemini_api_key,
                             temperature = 0.3)

@tool
def add_task(task, description=None):
    """Add a new task to the user's task list. Use this when the user wants to add or create a new task."""
    todoist.add_task(task, description=description)


@tool
def show_tasks():
    """Show all the user's tasks from Todoist. Use this tool when the user wants to see all their tasks."""
    print("here are your tasks:")
    results_paginator = todoist.get_tasks()
    tasks = []
    for task_list in results_paginator:
        for task in task_list:
            tasks.append(task.content)
    return tasks

tools = [add_task, show_tasks]
system_prompt = ("You are a helpful assistant. You will help the user add and view tasks. When asked to show the "
                 "tasks, please format them in a bullet list")
prompt = ChatPromptTemplate([("system", system_prompt),
                             MessagesPlaceholder("history"),
                             ("user", "{input}"),
                             MessagesPlaceholder("agent_scratchpad")])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

history = []
while True:
    user_input = input("You: ")
    response = agent_executor.invoke({"input": user_input, "history": history})
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response['output']))
