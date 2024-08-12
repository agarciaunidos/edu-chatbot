from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

from chat_retrieval import chat_agent, tools

def run_chat_interface():
    st.header("Policy Chat App")

    with st.sidebar:
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True,
                                          memory_key="chat_history", output_key="output")

        if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
            msgs.clear()
            st.session_state['steps'] = {}

    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            render_intermediate_steps(idx, msg, st.session_state.get('steps', {}))
            st.write(msg.content)

    if prompt := st.chat_input(placeholder="What is UnidosUS?"):
        st.chat_message("user").write(prompt)
        response = execute_chat_agent(prompt, memory)
        if response:
            display_chat_response(response, msgs)

def render_intermediate_steps(index, message, steps):
    """Utility to render intermediate steps saved in the session state."""
    for step in steps.get(str(index), []):
        if step[0].tool == "_Exception":
            continue
        with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
            st.write(step[0].log)
            st.write(step[1])

def execute_chat_agent(user_input, memory):
    """Executes the chat agent with the provided user input and returns the response."""
    agent_executor = AgentExecutor(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    try:
        return agent_executor.invoke({"input": user_input}, {"callbacks": [st_cb]})
    except Exception as e:
        st.error(f"An error occurred while executing the chat agent: {e}")
        return None

def display_chat_response(response, message_history):
    """Displays the chat agent's response and handles any necessary follow-up actions."""
    with st.chat_message("assistant"):
        st.write(response["output"])
        st.session_state['steps'][str(len(message_history.messages) - 1)] = response["intermediate_steps"]

def main():
    """Main entry point of the application."""
    try:
        run_chat_interface()
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()