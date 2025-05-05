import faiss
import numpy as np

# 1) Load your index
index = faiss.read_index(r"C:\Users\Devansh\Downloads\IITMandiHackathon-Group54\data (json+index+raw csv)\faiss.indexx")

# 2) Inspect properties
print("Total vectors in index:", index.ntotal)
d = index.d  # dimension
print("Embedding dimension:", d)

# If it's an IP index, make sure it's HNSW
print("Is HNSW?:", isinstance(index, faiss.IndexHNSWFlat))

# 3) Do a test search
#    a) Either load a real embedding (from embeddings.npy):
# embeddings = np.load("embeddings.npy")
# query = embeddings[0:1]   # first vector
#
#    b) Or use a random vector (normalized if IP index):
query = np.random.randn(1, d).astype("float32")
if index.metric_type == faiss.METRIC_INNER_PRODUCT:
    faiss.normalize_L2(query)

# 4) Search for the top-5 nearest neighbors
k = 5
distances, indices = index.search(query, k)
print(f"\nTop {k} results for test query:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
    print(f" {rank}. id={idx}, score={dist}")

# Check for any -1 (missing) indices or infinities
print("\nAny missing IDs?:", np.any(indices < 0))
print("Any infinite distances?:", np.isinf(distances).any())
