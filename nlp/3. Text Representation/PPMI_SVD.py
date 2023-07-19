import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Example documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Fit the documents and transform into count matrix
count_matrix = vectorizer.fit_transform(documents)

# Compute the co-occurrence matrix
co_occurrence_matrix = (count_matrix.T @ count_matrix).toarray()

# Compute the PPMI matrix
total_occurrences = np.sum(co_occurrence_matrix)
total_documents = count_matrix.shape[0]

ppmi_matrix = np.zeros(co_occurrence_matrix.shape, dtype=np.float32)
for i in range(co_occurrence_matrix.shape[0]):
    for j in range(co_occurrence_matrix.shape[1]):
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     pmi = np.log2((co_occurrence_matrix[i][j] * total_occurrences) /
        #                   (np.sum(co_occurrence_matrix[i]) * np.sum(co_occurrence_matrix[:, j])))
        
        # pmi = np.nan_to_num(pmi, posinf=0.0)  # 将NaN值替换为0
        # ppmi = max(pmi, 0)
        # ppmi_matrix[i][j] = ppmi
        
        p_xi = np.sum(co_occurrence_matrix[i])
        p_xj = np.sum(co_occurrence_matrix[:, j])
        p_xixj = co_occurrence_matrix[i][j]
        pmi = np.log2((p_xixj * total_occurrences) / (p_xi * p_xj))
        ppmi = max(pmi, 0) if pmi != 0 else 0
        ppmi_matrix[i][j] = ppmi

print("PPMI Matrix:")
np.savetxt('output.txt', ppmi_matrix, fmt='%.6f', delimiter=' ')

# 您可以取消注释第16到26行的代码，使用np.errstate来处理警告。或者，您可以保留第16到26行的注释，并使用第28到39行的代码来计算PPMI矩阵。


import numpy as np
from sklearn.decomposition import TruncatedSVD

# Load the PPMI matrix from the file
ppmi_matrix = np.loadtxt('output.txt')

# Create an instance of TruncatedSVD
svd = TruncatedSVD(n_components=2)

# Fit and transform the PPMI matrix using SVD
embeddings = svd.fit_transform(ppmi_matrix)

# Print the dense word embeddings
print("Word Embeddings:")
print(embeddings)