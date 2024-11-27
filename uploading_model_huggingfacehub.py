from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Path to the folder containing model files
model_path = "/home/chijofareen/Gomycode_projects/customer_feedback_analysis/summarization_model"

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Push the model to the Hub
model.push_to_hub("finetuned-BART_model")

# Push the tokenizer to the Hub
tokenizer.push_to_hub("finetuned-BART_model")
