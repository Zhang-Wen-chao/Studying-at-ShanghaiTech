from sklearn.feature_extraction.text import TfidfVectorizer

# Example documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create an instance of TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (terms)
feature_names = vectorizer.vocabulary_.keys()
# Print the TF-IDF scores for each document and term
for i in range(len(documents)):
    print(f"Document {i+1}:")
    for j, term in enumerate(feature_names):
        tfidf_score = tfidf_matrix[i, j]
        print(f"  {term}: {tfidf_score:.3f}")
    print()
