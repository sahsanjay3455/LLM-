import streamlit as st
from chat_bot import workflow

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Display previous messages
for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type here")

if user_input:

    # Store user message
    st.session_state['message_history'].append(
        {"role": "user", "content": user_input}
    )

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    config1 = {"configurable": {"thread_id": "thread-1"}}

    # IMPORTANT: Correct state format
    initial_state = {
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk, metadata in workflow.stream(
            initial_state,
            config=config1,
            stream_mode="messages"
        ):
            if hasattr(chunk, "content") and chunk.content:
                full_response += chunk.content
                placeholder.markdown(full_response)

    # Store assistant response
    st.session_state['message_history'].append(
        {"role": "assistant", "content": full_response}
    )