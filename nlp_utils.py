import nltk
from nltk.tokenize import word_tokenize

# Download 'punkt' tokenizer data (needed for word_tokenize)
nltk.download('punkt')

"""def process_text(text):
    tokens = word_tokenize(text.lower())
    return list(set(tokens))"""

def process_text(text):
    # Convert comma-separated string into clean symptom list
    return [s.strip().lower() for s in text.split(",") if s.strip()]

