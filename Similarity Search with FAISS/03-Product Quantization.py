import faiss
import numpy as np

# Step 1: Dataset
d = 128
nb = 100000
nq = 5

xb = np.random.random((nb, d)).astype("float32")
xq = np.random.random((nq, d)).astype("float32")

# Step 2: PQ Parameters
m = 8          # Number of subspaces
bits = 8       # Bits per codebook index

# Step 3: Create PQ Index
index = faiss.IndexPQ(d, m, bits)

# Step 4: Train PQ
index.train(xb)

# Step 5: Add vectors
index.add(xb)

# Step 6: Search
k = 5

D, I = index.search(xq, k)

print(I)
print(D)




# FLOW :

# Vectors
#     ↓
# Split into sub-vectors
#     ↓
# Learn codebooks
#     ↓
# Compress vectors
#     ↓
# Store PQ codes
#     ↓
# Query
#     ↓
# Approximate distance
# using lookup tables