"""LLM, graph, and tools setup"""
import os
from typing import Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEmbeddings
from typing_extensions import TypedDict

# Avoid circular import - document_retrieval will be added later

# Rebuild model to avoid Pydantic errors
ChatGoogleGenerativeAI.model_rebuild()


# === State ===
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize embedding function
embedding_fn = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'batch_size': 128, 'normalize_embeddings': True}
)

# Initialize tools (document_retrieval will be added after document_handling loads)
tools = [
    Tool(name="duckduckgo_search", func=DuckDuckGoSearchRun()._run, description="Search the web."),
    Tool(name="add", func=lambda a, b: a + b, description="Add two numbers."),
    Tool(name="multiply", func=lambda a, b: a * b, description="Multiply two numbers."),
]

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.5,
    google_api_key=os.environ["GOOGLE_API_KEY"],
)
llm_with_tools = llm.bind_tools(tools)

# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", lambda state: {"messages": [llm_with_tools.invoke(state["messages"])]})
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)


def create_conversation_graph(checkpointer):
    """Create a compiled graph with a checkpointer"""
    return graph_builder.compile(checkpointer=checkpointer)


def add_document_retrieval_tool(document_retrieval_func):
    """Add document retrieval tool to the tools list (only if not already added)"""
    # Check if tool already exists
    tool_names = [tool.name for tool in tools]
    if "document_retrieval" in tool_names:
        return  # Tool already added, skip
    
    tools.append(
        Tool(
            name="document_retrieval",
            func=document_retrieval_func,
            description="Retrieve relevant information from uploaded PDF documents. Use this when the user asks about content from PDFs they have uploaded, such as asking about specific problems, sections, or information from the documents."
        )
    )
    # Rebind LLM with updated tools
    global llm_with_tools
    llm_with_tools = llm.bind_tools(tools)

