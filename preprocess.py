import spacy
import nltk
from nltk.tokenize import word_tokenize

nlp = spacy.load("en_core_web_sm")
def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()
    tokens = word_tokenize(text)
    trimmed_text = " ".join(tokens)
    doc = nlp(trimmed_text)

    # Remove punctuation
    text = " ".join([word for word in[token.text for token in doc if not token.is_stop and not token.is_punct]])

    return text


print(preprocess_text("This is the life"))