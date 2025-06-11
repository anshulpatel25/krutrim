import asyncio
import os
import streamlit as st
import uuid

from dataclasses import dataclass
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from typing import Any

st.set_page_config(page_title="कृत्रिम", page_icon="🧠")


@dataclass
class Role:
    avatar: str
    name: str


rpi_mcp_url = os.getenv("KRUTRIM_RPI_MCP_URL")

client = MultiServerMCPClient(
    {
        "ambience": {
            "url": rpi_mcp_url,
            "transport": "streamable_http",
        }
    }
)


async def bot(state: MessagesState, config: RunnableConfig):

    tools = await client.get_tools()

    llm = ChatOllama(model="llama3.1", temperature=0.2)
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = SystemMessage(
        "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
    )

    return {
        "messages": llm_with_tools.invoke(
            [system_prompt] + state["messages"], config=config
        )
    }


def determine_role(message, human_role, assistant_role) -> Role:
    if type(message) == HumanMessage:
        return human_role
    else:
        return assistant_role


async def create_graph():

    tools = await client.get_tools()

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", bot)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


async def main():
    st.title("कृत्रिम - Simple AI Chatbot/Agent with MCP, designed in અમદાવાદ with ❤️")
    st.subheader(
        "Powered by Streamlit, LangGraph, LangChain, and Llama3.1", divider=True
    )

    graph: CompiledStateGraph = await create_graph()
    human_role: Role = Role(avatar="👱🏽", name="user")
    assistant_role: Role = Role(avatar="🤖", name="assistant")
    config: dict[str, Any] = {"configurable": {"thread_id": uuid.uuid1()}}

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        current_role = determine_role(
            message, human_role=human_role, assistant_role=assistant_role
        )
        with st.chat_message(name=current_role.name, avatar=current_role.avatar):
            st.markdown(message.content)

    if prompt := st.chat_input("How can I help you?"):
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message(name=human_role.name, avatar=human_role.avatar):
            st.markdown(prompt)

        with st.chat_message(name=assistant_role.name, avatar=assistant_role.avatar):
            with st.spinner("⏱️", show_time=True):
                response = await graph.ainvoke(
                    {"messages": st.session_state.messages}, config=config
                )
                st.markdown(response["messages"][-1].content)
                st.markdown(
                    f"Total Tokens Used: **{response["messages"][-1].usage_metadata["total_tokens"]}**"
                )

        st.session_state.messages.append(response["messages"][-1])


if __name__ == "__main__":
    asyncio.run(main())
