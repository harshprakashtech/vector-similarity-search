import numpy as np

from data import data
from core import vector_search


# Search query
search_query = np.array([0.88, 0.03, 0.47, 0.29, 0.11, 0.68, 0.26, 0.82]) # Closer to "animal"


# Vector similarity search
result = vector_search(search_query, data, 3)

print(result)