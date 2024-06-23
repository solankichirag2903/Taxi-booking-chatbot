import os
import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Set your Huggingface API token
os.environ['HUGGINGFACE_API_TOKEN'] = 'Your-HuggingFace-Api'

# Define the model ID
model_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

# Initialize the Hugging Face Hub model
custom_llm = HuggingFaceEndpoint(
    huggingfacehub_api_token=os.environ['HUGGINGFACE_API_TOKEN'],
    repo_id=model_id,
    temperature=0.5,
    max_new_tokens=2000
)

# Define the template for conversation
template = """
User: {user_input}
taxibooking bot:
"""

# Create the PromptTemplate and LLMChain
prompt = PromptTemplate(template=template, input_variables=['user_input'])
customllm_chain = LLMChain(llm=custom_llm, prompt=prompt)

def main():
    # Streamlit app layout
    st.title("Taxi Booking Chatbot")
    st.write("Hello! Welcome to the Taxi booking service. Say 'Hi' to start the conversation.")

    # Initialize conversation messages
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {'role': 'assistant', 'content': 'Hi, I am Taxi Booking Bot. How can I assist you today?'}
        ]

    # User input
    user_input = st.text_input("You: ", key="input")

    # Handle user input
    if st.button("Submit") and user_input.strip():
        # Add user message to conversation
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        
        # Generate response using the Mistral model
        response = customllm_chain.run(user_input=user_input)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    # Display alternating conversation history
    st.subheader("Conversation History")
    for i in range(len(st.session_state.messages)):
        message = st.session_state.messages[i]
        if message['role'] == 'user':
            st.text_area("You:", value=message['content'], height=80, key=f"user_{i}")
        elif message['role'] == 'assistant':
            st.text_area("Bot:", value=message['content'], height=80, key=f"bot_{i}")

if __name__ == "__main__":
    main()
