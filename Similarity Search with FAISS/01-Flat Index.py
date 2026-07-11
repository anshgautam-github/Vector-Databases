import faiss
import numpy as np


# Step 1: Create sample vectors
d = 128                     # Vector dimension
nb = 10000                  # Number of database vectors
nq = 5                      # Number of queries

xb = np.random.random((nb, d)).astype("float32")
xq = np.random.random((nq, d)).astype("float32")


# Step 2: Create Flat Index
index = faiss.IndexFlatL2(d)

# Step 3: Add vectors
index.add(xb)

# Step 4: Search
k = 5
D, I = index.search(xq, k)

print(I)
print(D)



# FLOW :

# Vectors
#     ↓
# Create Flat Index
#     ↓
# Add vectors
#     ↓
# Query
#     ↓
# Compare against ALL vectors
#     ↓
# Return Top-k