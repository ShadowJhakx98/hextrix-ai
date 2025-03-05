import nltk

nltk.download('punkt') # Ensure punkt tokenizer is downloaded
nltk.download('punkt_tab') # Download punkt_tab specifically (as per error message)
nltk.download('stopwords') # Ensure stopwords data is downloaded
nltk.download('vader_lexicon') # Ensure VADER lexicon is downloaded
nltk.download('averaged_perceptron_tagger') # Download averaged_perceptron_tagger - often needed for tokenizers and taggers

print("NLTK data download complete. Please restart your Flask app.")