import faiss
import numpy as np

# Step 1: Dataset
d = 128
nb = 100000
nq = 5

xb = np.random.random((nb, d)).astype("float32")
xq = np.random.random((nq, d)).astype("float32")

# Step 2: Parameters
nlist = 100
k = 5

# Step 3: Quantizer
quantizer = faiss.IndexFlatL2(d)

# Step 4: Create IVF Index
index = faiss.IndexIVFFlat(
    quantizer,
    d,
    nlist
)

# Step 5: Train (learn centroids)
index.train(xb)

# Step 6: Add vectors
index.add(xb)

# Step 7: Search
index.nprobe = 10

D, I = index.search(xq, k)

print(I)
print(D)