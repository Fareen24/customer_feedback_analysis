
# Customer Feedback Analysis and Summarization

This project is designed to help businesses gain actionable insights from customer feedback collected across multiple sources, such as product reviews, social media comments, and surveys. By leveraging NLP techniques and a summarization model (BART), the tool will analyze and summarize customer feedback, detect themes and sentiments, and generate a streamlined summary of qualitative feedback data.

## Project Structure

The project is structured into three main phases:

1. **Data Preparation and Preprocessing** 
2. **Fine-tuning BART for Text Summarization**
3. **Building a Knowledge-Augmented Chatbot Using LlamaIndex**

## Data Files

The following dataset files are hosted on Google Drive due to size limitations on GitHub:


- [curated_reviews_data.csv](https://drive.google.com/file/d/1dmmQtqKU3WA74_cK8oBZa4okVhfQQIJs/view?usp=sharing)
- [processed_subset_data.csv](https://drive.google.com/file/d/1fdyqniQoO2PHI96ipob-_ef3YczkAwiG/view?usp=sharing) - This dataset has been preprocessed, has extracted themes and sentiments and tokenized to be used for the second phase, fine-tuning using Bart.
- datasets: Located in the `datasets` folder on [Google Drive](https://drive.google.com/drive/folders/1pNJE9kMc3oGqJqHx9ZOowiB7LY0SFUhK?usp=sharing). - In the folder is three datasets that were used to curate the dataset `curated_reviews_data` to be used in the Data Preparation and Preprocessing phase.

To use these datasets, download them and place them in the appropriate folder in your local project directory.


## Phase 1: Data Preparation and Preprocessing 

In this phase, we focused on cleaning, preprocessing, and preparing the customer feedback data for model training.

### Steps Completed

1. **Data Loading and Inspection**
   - Loaded customer feedback from multiple sources, such as product reviews, into a Pandas DataFrame named `data`.
   - Inspected the data for any inconsistencies and missing values.

2. **Preprocessing Text Data**
   - Created a text preprocessing pipeline to clean the review text and titles, including:
     - **Lowercasing** all text to standardize format.
     - **Removing punctuation and special characters** to reduce noise.
     - **Removing stopwords** with NLTK to focus on meaningful words.
     - **Tokenization** using NLTK to break down text into individual tokens.
     - **Lemmatization** to reduce words to their root form.
   - Saved the cleaned data as `clean_review_text` and `clean_review_title` in the DataFrame.

3. **Sentiment Analysis and Theme Detection**
   - Conducted sentiment analysis using the VADER sentiment analysis tool to label feedback as positive, negative, or neutral.
   - Applied LDA topic modeling to extract key themes from the reviews, providing insights into common customer issues and highlights.

4. **Tokenization with Hugging Face Transformers**
   - Used Hugging Face’s `BartTokenizer` to tokenize the clean text for BART, preparing it for summarization tasks in the next phase.

5. **Saving Processed Data**
   - Saved the processed data as a CSV file for easy loading in future phases.

## Phase 2: Fine-Tuning BART for Text Summarization

In this phase, we will fine-tune a BART model to generate summaries from customer feedback, focusing on extracting concise insights. 

### Steps Overview
1. **Prepare Training Data**: Create input-output pairs for training BART using `full_text` as input and providing for generating target summaries.
2. **Fine-Tune BART**: Use Hugging Face Transformers to fine-tune BART, adjusting parameters like learning rate and batch size.
3. **Evaluation**: Measure performance using summarization metrics like  BLEU, as well as human evaluations where possible.

## Phase 3: Building a Knowledge-Augmented Chatbot Using LlamaIndex

In this phase, the summarization model will be paired with a Retrieval-Augmented Generation (RAG) system using LlamaIndex to create a chatbot that provides insights from feedback and retrieves additional context from curated knowledge bases.



---
