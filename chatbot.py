import os
import torch
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests


# Model paths and IDs
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
bart_model_path = "ChijoTheDatascientist/finetuned-BART_model"
knowledge_base_url = "https://raw.githubusercontent.com/Fareen24/customer_feedback_analysis/main/data/data.txt"

# Load BART model for summarization 
device = torch.device('cpu') 

# Initialize BART tokenizer and model
bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_path)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_model_path).to(device)

# Function to summarize reviews using BART
def summarize_review(review_text):
    inputs = bart_tokenizer(review_text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = bart_model.generate(inputs["input_ids"], max_length=40, min_length=10, length_penalty=2.0, num_beams=8, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load the HuggingFace model for response generation
def get_llm_hf_inference(model_id=model_id, max_new_tokens=128, temperature=0.1):
    hf_model = HuggingFaceEndpoint(
        repo_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token=os.getenv("HF_TOKEN")  
    )
    return hf_model

# Load knowledge base text from GitHub raw URL
def load_knowledge_base(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.text
    except Exception as e:
        return f"Error loading knowledge base: {e}"

knowledge_base = load_knowledge_base(knowledge_base_url)

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
        response = chat.invoke(input={
            "system_message": system_message,
            "knowledge_base": knowledge_base,
            "chat_history": chat_history,
            "user_input": user_input
        })
        response = response.split("AI:")[-1]
        return response
    except Exception as e:
        return f"Error generating response: {e}"

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
            
            # Update chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.markdown(f"**Response:** \n{response}")
    else:
        st.warning("Please enter a query.")
