import os
import torch
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
import time
from huggingface_hub import login

# Access the Hugging Face token from Streamlit secrets
hf_token = st.secrets["huggingface"]["HF_TOKEN"]

# Model paths and IDs
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
bart_model_path = "ChijoTheDatascientist/summarization-model"
knowledge_base_url = "https://raw.githubusercontent.com/Fareen24/customer_feedback_analysis/main/data/data.txt"

# Load BART model for summarization 
device = torch.device('cpu') 

# Load model and tokenizer directly with authentication
bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_path)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_model_path).to(device)

# Function to summarize reviews using BART
@st.cache_data
def summarize_review(review_text):
    inputs = bart_tokenizer(review_text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = bart_model.generate(inputs["input_ids"], max_length=40, min_length=10, length_penalty=2.0, num_beams=8, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load the HuggingFace model for response generation
def get_llm_hf_inference(model_id=model_id, max_new_tokens=128, temperature=0.1):
    # Use HuggingFaceEndpoint and pass the token
    hf_model = HuggingFaceEndpoint(
        repo_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token=hf_token  
    )
    return hf_model

@st.cache_data
def generate_hf_response(system_message, user_input, chat_history):
    hf_model = get_llm_hf_inference()
    prompt = PromptTemplate.from_template(
        (
            "[INST] {system_message}"
            "\nKnowledge Base:\n{knowledge_base}"
            "\nConversation History:\n{chat_history}\n\n"
            "User: {user_input}\n[/INST]\nAI:"
        )
    )
    chat = prompt | hf_model.bind(skip_prompt=True) | StrOutputParser(output_key='content')
    response = chat.invoke(input={
        "system_message": system_message,
        "knowledge_base": knowledge_base,
        "chat_history": chat_history,
        "user_input": user_input
    })
    response = response.split("AI:")[-1]
    return response

# Load knowledge base text from GitHub raw URL
def load_knowledge_base(url):
    try:
        response = requests.get(url)
        response.raise_for_status() 
        return response.text
    except Exception as e:
        return f"Error loading knowledge base: {e}"

knowledge_base = load_knowledge_base(knowledge_base_url)

# Handling rate limits
def request_with_retry(url, headers, retries=5, delay=60):
    for _ in range(retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  
            return response
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print("Rate limit exceeded, retrying...")
                time.sleep(delay)  
            else:
                raise e
    return None

# Chat function to generate responses 
def generate_response(system_message, user_input, chat_history, max_new_tokens=128):
    try:
        hf_model = get_llm_hf_inference(max_new_tokens=max_new_tokens)
        prompt = PromptTemplate.from_template(
            (
                "[INST] {system_message}"
                "\nKnowledge Base:\n{knowledge_base}"
                "\nConversation History:\n{chat_history}\n\n"
                "User: {user_input}\n[/INST]\nAI:"
            )
        )
        chat = prompt | hf_model.bind(skip_prompt=True) | StrOutputParser(output_key='content')

        # Attempt to invoke API request
        response = chat.invoke(input={
            "system_message": system_message,
            "knowledge_base": knowledge_base,
            "chat_history": chat_history,
            "user_input": user_input
        })

        response = response.split("AI:")[-1]
        return response

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return "You have reached your limit... try again later"
        else:
            return f"An error occurred: {e}"
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit app configuration
st.set_page_config(page_title="Insight Snap")
st.title("Insight Snap")

st.markdown("""
- Use specific keywords in your queries to get targeted responses:
    - **"summarize"**: To provide summary of your customers review.
    - **"Feedback or insights"**: Get actionable business insights based on feedback.
""")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat interface 
user_input = st.text_input("Enter your query:")

if st.button("Submit"):
    if user_input:
        # Summarize if the query contains the word 'summarize'
        if "summarize" in user_input.lower():
            summary = summarize_review(user_input)
            st.markdown(f"**Summary:** \n{summary}")
        else:
            # Generate response using HuggingFace model for business insights
            system_message = "You are a helpful assistant providing insights from customer feedback."
            response = generate_response(system_message, user_input, st.session_state.chat_history)
            
            # Handle empty or error responses 
            if response:
                # Update chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.markdown(f"**Response:** \n{response}")
            else:
                st.warning("No response generated. Please try again later.")
    else:
        st.warning("Ask me a question on customer feedback...")
