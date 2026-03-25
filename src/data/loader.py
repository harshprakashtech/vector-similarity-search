# import numpy as np

# Fake vector data
# data = {
#     "bicycle":  np.array([0.21, 0.58, 0.44, 0.12, 0.87, 0.35, 0.79, 0.01]),
#     "lion":     np.array([0.91, 0.02, 0.45, 0.31, 0.11, 0.67, 0.23, 0.84]),
#     "airplane": np.array([0.15, 0.92, 0.63, 0.05, 0.77, 0.41, 0.50, 0.12]),
#     "motorcycle": np.array([0.25, 0.55, 0.40, 0.15, 0.80, 0.30, 0.70, 0.05]),
#     "tiger": np.array([0.85, 0.05, 0.50, 0.28, 0.12, 0.70, 0.30, 0.80])
# }

# Generate real vector data using "sentence transformers" embedding model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2") # Define model

# Define sample sentences for data
sentences = [
    "a dog playing in the park",
    "a cat sleeping on the couch",
    "a motorcycle speeding on the highway",
    "a bicycle parked near the school",
    "a lion hunting in the savanna",
    "a tiger hiding in the jungle",
    "an airplane flying above the clouds",
]

# Generate embeddings for each sentence
data = {sentence: model.encode(sentence) for sentence in sentences}