import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "Hello, world! How are you?"
tokens = word_tokenize(text)
print(tokens)
