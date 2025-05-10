from typing import Annotated, TypedDict, Any
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from dataclasses import dataclass
import uuid
import streamlit as st

st.set_page_config(page_title="कृत्रिम", page_icon="🧠")


class State(TypedDict):
    messages: Annotated[list, add_messages]


@dataclass
class Role:
    avatar: str
    name: str


def chatbot(state: State) -> dict[str, BaseMessage]:
    llm = init_chat_model(model="gemma3", model_provider="ollama")

    state["messages"].append(
        SystemMessage(
            content="You are a helpful assistant that provides information and answers questions correctly, if you don't know the answer, say 'I don't know'."
        )
    )

    return {"messages": llm.invoke(state["messages"])}


def determine_role(message, human_role, assistant_role) -> Role:
    if type(message) == HumanMessage:
        return human_role
    else:
        return assistant_role


def create_graph() -> CompiledStateGraph:
    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def main():
    st.title("कृत्रिम - Simple AI Chatbot, designed in અમદાવાદ with ❤️")
    st.subheader(
        "Powered by Streamlit, LangGraph, LangChain, and Google Gemma3", divider=True
    )

    bot: CompiledStateGraph = create_graph()
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
            response = bot.invoke(
                {"messages": st.session_state.messages}, config=config
            )
            st.markdown(response["messages"][-1].content)
            st.markdown(
                f"Total Tokens Used: **{response["messages"][-1].usage_metadata["total_tokens"]}**"
            )

        st.session_state.messages.append(response["messages"][-1])


if __name__ == "__main__":
    main()
