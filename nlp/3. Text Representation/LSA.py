from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Example documents
documents = [
    "I love coding",
    "Coding is fun",
    "Python is awesome",
    "I love Python"
]

# Create an instance of TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents into TF-IDF matrix
tf_idf_matrix = vectorizer.fit_transform(documents)

# Create an instance of TruncatedSVD
lsa = TruncatedSVD(n_components=2)

# Fit and transform the TF-IDF matrix into LSA matrix
lsa_matrix = lsa.fit_transform(tf_idf_matrix)

# Print the LSA matrix
print("LSA Matrix:")
print(lsa_matrix)