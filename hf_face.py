import os

os.environ['HF_HOME'] = '/Users/utkarshshukla/Downloads/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/Users/utkarshshukla/Downloads/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/Users/utkarshshukla/Downloads/huggingface_cache'

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentences = [
    "I love playing cricket",
    "Python is a programming language",
    "Delhi is the capital of India",
    "I enjoy eating pizza"
]

query = "What is largest city in india?"

sentence_embeddings = model.encode(sentences)
query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, sentence_embeddings)
scores = similarities[0]

best_match_index = scores.argmax()

# Print results
print("Query:", query)
print()

for i, sentence in enumerate(sentences):
    print(f"{sentence} --> Score: {scores[i]:.4f}")

print("\nBest Match:")
print(sentences[best_match_index])