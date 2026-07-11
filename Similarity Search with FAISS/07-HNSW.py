import faiss
import numpy as np

# Step 1: Dataset
d = 128
nb = 100000
nq = 5

xb = np.random.random((nb, d)).astype("float32")
xq = np.random.random((nq, d)).astype("float32")

# Step 2: Parameters
M = 16

index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = 40
index.hnsw.efSearch = 16

# Step 3: Build graph
index.add(xb)

# Step 4: Search
k = 5

D, I = index.search(xq, k)

print(I)
print(D)



# Vectors
#       ↓
# Create Empty Graph
#       ↓
# Insert vectors
#       ↓
# Build graph automatically
#       ↓
# Query
#       ↓
# Navigate graph
#       ↓
# Return Top-k