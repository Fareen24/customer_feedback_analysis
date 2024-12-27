
# Customer Feedback Analysis and Summarization


This chatbot provides actionable insights from customer feedback collected across various sources, such as product reviews from Sephora, Amazon and an online clothing store. It leverages abstractive summarization using BART Transformers and advanced natural language processing capabilities from the Mistral model to analyze feedback, detect themes and sentiments, and generate streamlined summaries and actionable insights.

🌐 [Try the App on Hugging Face Spaces](https://huggingface.co/spaces/ChijoTheDatascientist/chatbot-customer-feedback)


## Project Structure

The project is structured into three main phases:

1. **Data Preparation and Preprocessing** 
2. **Fine-tuning BART for Text Summarization**
3. **Used HuggingFace Mistral Model for generating business insights and feedback**

# Project files
## Data Files

The following dataset files are hosted on Google Drive due to size limitations on GitHub:


- [curated_reviews_data.csv](https://drive.google.com/file/d/1dmmQtqKU3WA74_cK8oBZa4okVhfQQIJs/view?usp=sharing)
- [processed_subset_data.csv](https://drive.google.com/file/d/1fdyqniQoO2PHI96ipob-_ef3YczkAwiG/view?usp=sharing) - This dataset has been preprocessed, has extracted themes and sentiments and tokenized to be used for the second phase, fine-tuning using Bart.
- datasets: Located in the `datasets` folder on [Google Drive](https://drive.google.com/drive/folders/1pNJE9kMc3oGqJqHx9ZOowiB7LY0SFUhK?usp=sharing). - In the folder is three datasets that were used to curate the dataset `curated_reviews_data` to be used in the Data Preparation and Preprocessing phase.

To use these datasets, download them and place them in the appropriate folder in your local project directory.

## Other files
**data_preprocessing.ipynb**: Notebook on data preparation and preprocessing
**train_validation_splitting.ipynb**: Notebook on splitting the data into train, test and validation sets for fine-tuning the BART Model
**finetuning_with_BART**: Notebook on finetuning BART Model and saving the model and tokenizer.
**uploading_model_huggingfacehub.py**: python script of uploading finetuned model to cloud on hugging face space.

**application file**: The applcation file is hosted on the hugging face platform where the app is hosted.

## Phase 1: Data Preparation and Preprocessing 

In this phase, we focused on cleaning, preprocessing, and preparing the customer feedback data for model training.

### Steps taken

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
   - Saved the cleaned data  in the DataFrame.

3. **Sentiment Analysis and Theme Detection**
   - Conducted sentiment analysis using the VADER sentiment analysis tool to label feedback as positive, negative, or neutral.
   - Applied LDA topic modeling to extract key themes from the reviews, providing insights into common customer issues and highlights.

4. **Tokenization with Hugging Face Transformers**
   - Used Hugging Face’s `BartTokenizer` to tokenize the clean text for BART, preparing it for summarization tasks in the next phase.

5. **Saving Processed Data**
   - Saved the processed data as a CSV file for easy loading in future phases.

## Phase 2: Fine-Tuning BART for Text Summarization

### Steps taken
In this phase, I  split the data into training, validation, and test datasets and converting it into a huggung face dataset for easy copatibility during fine-tuning of the model. Then saved it to load it for finetuning.

Then using the `clean_review_text` as input and `clean_summary` as the target toeknized the dataset  tokenized it  and used it for fine-tuning the BART model.

On the dataset we compared summaries that the model generated with the one from the dataset. 


This is in the `train_validation_splitting` notebook and `finetuning_with_BART` notebook.

## Phase 3: Building the Chatbot
The chatbot integrated two key models:

1. **Fine-tuned BART model** for summarizing customer reviews that was uploaded on the hugging face spaces.
2. **HuggingFace Mistral 7B Model** for generating business insights and feedback.

## Features
The chatbot is able to:

1. Summarize reviews
2. Give insights on customer feedback and general queries about it.

