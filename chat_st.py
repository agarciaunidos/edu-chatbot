import streamlit as st
import time
import pandas as pd
from uuid import uuid4
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from aws_secrets_initialization import dynamodb_history
from chat_retrieval import chat_agent, tools

# Initialize session state
def initialize_session_state():
    """
    Initialize the session state with default values.
    """
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid4())
    
    st.session_state.setdefault("messages", [{"role": "user", "content": "UUS Assistant"}])
    st.session_state.setdefault("prompt", None)
    st.session_state.setdefault("feedback_form", False)
    st.session_state.setdefault("feedback_value", 3)
    st.session_state.setdefault("feedback_text", "")


# Handle user feedback
def handle_feedback():

    feedback_value = st.session_state["feedback_value"]
    feedback_text = st.session_state["feedback_text"]
    """
    Process and store user feedback in DynamoDB.
    """
    try:
        timestamp = int(time.time())
        dynamodb_history.add_message(SystemMessage(
            id='1',
            user_id='test_user',
            content='feedback',
            st_session_id=st.session_state["session_id"],
            response_metadata={
                'timestamp': timestamp,
                'face_feedback': feedback_value,
                'feedback_text': feedback_text,
                'score': feedback_value,
            }
        ))
        st.success("Thank you for your feedback!")
        
        #st.session_state.feedback_submitted = True
    except Exception as e:
        st.error(f"An error occurred while handling feedback: {e}")

# Format search results as a DataFrame
def format_search_results_as_dataframe(documents):
    """
    Convert search results to a pandas DataFrame.
    
    Args:
    documents (list): List of document objects with metadata

    Returns:
    pd.DataFrame: Formatted search results
    """
    results = []
    for document in documents:
        metadata = document.metadata
        source_url = metadata.get('source', '').replace('s3://', 'https://s3.amazonaws.com/')
        relevance_score = metadata.get('relevance_score', 0) * 100
        results.append({
            "Title": metadata.get('title', ''),
            "Page": metadata.get('page', ''),
            "Source": source_url,
            "Relevance Score": relevance_score
        })
    return pd.DataFrame(results)

# Render intermediate steps
def render_intermediate_steps(index, message, steps):
    """
    Render intermediate steps of the chat agent.
    
    Args:
    index (int): Step index
    message (object): Message object
    steps (dict): Dictionary of steps
    """
    for step in steps.get(str(index), []):
        if step[0].tool == "_Exception":
            continue
        with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
            st.write(step[0].log)
            st.write(step[1])

# Execute chat agent
def execute_chat_agent(user_input, memory):
    """
    Execute the chat agent with the given input and memory.
    
    Args:
    user_input (str): User's input message
    memory (ConversationBufferMemory): Chat memory object

    Returns:
    dict: Chat agent's response
    """
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

# Display chat response
def display_chat_response(response, message_history):
    """
    Display the chat agent's response and sources.
    
    Args:
    response (dict): Chat agent's response
    message_history (StreamlitChatMessageHistory): Chat message history
    """
    with st.chat_message("assistant"):
        st.write(response["output"])
        try:
            sources = response["intermediate_steps"][0][1]
            df = format_search_results_as_dataframe(sources)
            st.subheader('Sources:')
            st.data_editor(df, column_config={
                "Source": st.column_config.LinkColumn("Source"),
                "Relevance Score": st.column_config.NumberColumn("Relevance Score", format='%.2f %%')
            }, hide_index=True)
        except IndexError:
            pass


# Main chat interface
def run_chat_interface():
    """
    Run the main chat interface using Streamlit.
    """
    st.header("Education Federal Student Aid Assistant App")

    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs,
        return_messages=True,
        memory_key="chat_history",
        output_key="output"
    )

    if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
        msgs.clear()
        st.session_state['steps'] = {}

    # Display chat history
    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            render_intermediate_steps(idx, msg, st.session_state.get('steps', {}))
            st.write(msg.content)

    # Handle user input
    if prompt := st.chat_input(placeholder="What is the Education Federal Student Aid?"):
        st.chat_message("user").write(prompt)
        dynamodb_history.add_user_message(HumanMessage(id=st.session_state['session_id'], content=prompt))
        
        response = execute_chat_agent(prompt, memory)
        
        if response:
            display_chat_response(response, msgs)
            dynamodb_history.add_ai_message(AIMessage(id=st.session_state['session_id'], content=response["output"]))
            st.session_state["feedback_form"] = True
            
            # Display feedback form
    if st.session_state["feedback_form"] == True:
        with st.form(key="feedback_form_st"):
            st.write("We'd love to hear your feedback!")
            feedback_value = st.slider(
                "How would you rate this response?", 
                min_value=1, max_value=5,
                key="feedback_slider"
            )
            feedback_text = st.text_area(
                "Any additional comments?", 
                placeholder="Please share your thoughts...",
                key="feedback_text_area"
            )
            #st.session_state.feedback_value = feedback_value
            #st.session_state.feedback_text = feedback_text
            submit_button = st.form_submit_button("Submit Feedback")
            
            if submit_button:
                st.session_state.feedback_value = feedback_value
                st.session_state.feedback_text = feedback_text
                handle_feedback()
                st.session_state["feedback_form"] = False
                st.rerun()


    # Handle feedback submission

# Main function
def main():
    """
    Main function to run the Streamlit app.
    """
    try:
        run_chat_interface()
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    initialize_session_state()
    main()