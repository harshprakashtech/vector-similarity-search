import numpy as np

# Helper function to calculate Cosine similarity of two vectors

# Dot product: a * b
# Magnitude of a, |a|: sqrt(a1^2 + a2^2 + ....)
# Magnitude of b, |b|: sqrt(b1^2 + b2^2 + ....)

# Cosine formula: (a * b)/|a||b| 

def cosine_similarity(query_vec: np.ndarray, candidate_vec: np.ndarray) -> float:
    # dot_product: float = np.sum(query_vec * candidate_vec)
    dot_product: float = np.dot(query_vec, candidate_vec)
    
    # magnitude_query_vec: float = np.sqrt(np.sum(query_vec ** 2))
    # magnitude_candidate_vec: float = np.sqrt(np.sum(candidate_vec ** 2))
    
    magnitude_query_vec: float = np.linalg.norm(query_vec)
    magnitude_candidate_vec: float = np.linalg.norm(candidate_vec)
    
    return dot_product/(magnitude_query_vec * magnitude_candidate_vec)