# Import necessary modules
from typing import List, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph import StateGraph, END
import functools
import operator

# Initialize the language model
llm = ChatOpenAI(api_key="")

# Define tools
search_tool = TavilySearchResults(max_results=5)
code_tool = PythonREPLTool()

# Function to create a worker agent
def create_worker_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt_template)
    return AgentExecutor(agent=agent, tools=tools)

# Define the agent state
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    next: str

# Function for an agent node
def agent_execution_node(state, agent, name):
    result = agent.invoke(state)
    state["messages"].append(HumanMessage(content=result["output"], name=name))
    return {"messages": state["messages"]}

# Create worker agents
tdm_agent = create_worker_agent(llm, [search_tool], "You are a technical delivery manager responsible for conducting technical interviews with clients. Summarize the conversation you had during an interview with a candidate")
tdm_node = functools.partial(agent_execution_node, agent=tdm_agent, name="TDM Agent")

# Define the supervisor agent
worker_names = ["TDM Agent"]
system_prompt = (
    "You are the supervisor managing the following workers: {worker_names}. "
    "Based on the user's request, decide which worker should handle the next task. "
    "Each worker will return their results and status. Respond with FINISH when done."
)
options = worker_names + ["FINISH"]
function_definition = {
    "name": "route",
    "description": "Determine the next worker.",
    "parameters": {
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "enum": options,
            }
        },
        "required": ["next"],
    },
}
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system", f"Who should act next? Choose from: {options}"),
]).partial(worker_names=", ".join(worker_names))
supervisor_chain = (supervisor_prompt | llm.bind_functions(functions=[function_definition], function_call="route") | JsonOutputFunctionsParser())

# Create the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("TDM Agent", tdm_node)
workflow.add_node("supervisor", supervisor_chain)

# Add edges to the workflow
for worker in worker_names:
    workflow.add_edge(worker, "supervisor")
conditional_map = {k: k for k in worker_names}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Set entry point and compile the workflow
workflow.set_entry_point("supervisor")
graph = workflow.compile()

# Test the multi-agent workflow
tasks = [
    "Please summarize this transcript: \nBasma Muhammad 19 minutes 48 seconds\nUh, what are hooks in react?\nJose\n19 minutes 52 seconds\nJose 19 minutes 52 seconds\nYeah, hooks is like a new feature that react introduced in in version 16.\nJose 19 minutes 59 seconds\nSo basically it's like a less verbose way to do things like for example the yeah, we have like a hook to to update and state.\nJose 20 minutes 11 seconds\nYeah.\nJose 20 minutes 12 seconds\nSo basically it's a function that allows you to update, update the state.\nJose 20 minutes 16 seconds\nWe have hooks for instead of having a constructor or a function like component did update and everything we have who call use effect that it's going to allow us to.\nJose 20 minutes 32 seconds\nTo execute, call at the beginning of the of the of the component load component and then we can just like.\nJose 20 minutes 41 seconds\nThe certain type of things and the when some type of when certain variables changes also execute this type of things.\nJose 20 minutes 50 seconds\nSo it it's like a less verbose way to create or manage different type of variables in in in react."
]

for task in tasks:
    for state in graph.stream({"messages": [HumanMessage(content=task)]}):
        if "__end__" not in state:
            print(state)
            print("----")
