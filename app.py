from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import openai
import streamlit as st

from dotenv import load_dotenv
import os
load_dotenv()

# Langsmith tracking
os.environ["LANGCHAIN_PROJECT"] = "Web Q&A Chatbot App"


# defining prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "{question}")
    ]
)


def generate_response(
        question: str, 
        api_key: str,
        provider: str,
        model: str, 
        temperature: float, 
        max_tokens: int
) -> str:
    """
    Generate response from the LLM model based on user question.

    Args:
        question (str): The user's question.
        api_key (str): OpenAI API key.
        provider (str): The LLM provider to use.
        model (str): The LLM model to use.
        temperature (float): Sampling temperature (closer to 0 means less random, closer to 1 means more random).
        max_tokens (int): Maximum tokens in the response.
    
    Returns:
        str: The generated response.
    """
    if provider.lower() == "openai":
        openai.api_key = api_key
        llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider.lower() == "ollama":
        llm = Ollama(model=model)
    
    output_parser = StrOutputParser()

    chain = prompt_template | llm | output_parser
    response = chain.invoke({"question": question})
    return response


# creating streamlit app
st.title("Web Q&A Chatbot App")
st.write("You can ask a question about anything!")

# side bar for model configurations
st.sidebar.title("Model Configuration")

# dropdown for provider selection
provider = st.sidebar.selectbox(
    "Select Provider",
    ("OpenAI", "Ollama", ""),
)

# dropdown for models based on the provider
if provider.lower() == "openai":
    # api key input for openai
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", ""]
else:       # ollama provider
    model_options = ["gemma:2b", ""]

model = st.sidebar.selectbox("Select Model", model_options)

# adjusting response parameter
temperature = st.sidebar.slider("Temperature",
                                min_value=0.0, max_value=1.0, value=0.7, step=0.1)
max_tokens = st.sidebar.slider("Max Tokens",
                                min_value=50, max_value=300, value=200, step=100)

# user interface for question input
st.write("Ask you question here: ")
user_input = st.text_input("User: ")

if st.button("Generate Response"):
    if not user_input:
        st.info("Please enter a question to get started")
    elif provider.lower() == "openai" and not api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar.")
    else:
        with st.spinner("Generating response..."):
            if provider.lower() == "ollama":
                api_key = None    # no api key needed for ollama
            
            response = generate_response(
                question=user_input,
                api_key=api_key,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        st.write(f"**Chatbot:** {response}")
