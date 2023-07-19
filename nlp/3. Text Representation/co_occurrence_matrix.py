import numpy as np
from collections import defaultdict
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')


def create_co_occurrence_matrix(corpus, window_size=2):
    co_occurrence_matrix = defaultdict(lambda: defaultdict(int))

    # Tokenize the corpus
    tokens = word_tokenize(corpus.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Iterate over the tokens to create the co-occurrence matrix
    for i in range(len(tokens)):
        target_word = tokens[i]
        for j in range(i - window_size, i + window_size + 1):
            if j != i and j >= 0 and j < len(tokens):
                co_occurrence_matrix[target_word][tokens[j]] += 1

    # Convert the co-occurrence matrix to a numpy array
    words = list(co_occurrence_matrix.keys())
    matrix = np.zeros((len(words), len(words)), dtype=np.int32)
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            matrix[i][j] = co_occurrence_matrix[word1][word2]

    return matrix, words

# Example usage
corpus = "I like to play football with my friends. My friends also enjoy playing football."
co_matrix, words = create_co_occurrence_matrix(corpus)

# Print the co-occurrence matrix
for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        print(f"Co-occurrence of '{word1}' and '{word2}': {co_matrix[i][j]}")
