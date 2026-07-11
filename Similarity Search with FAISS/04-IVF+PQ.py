import faiss
import numpy as np

d = 128
nb = 100000
nq = 5

xb = np.random.random((nb, d)).astype("float32")
xq = np.random.random((nq, d)).astype("float32")

nlist = 100
m = 8
bits = 8
k = 5

quantizer = faiss.IndexFlatL2(d)

index = faiss.IndexIVFPQ(
    quantizer,
    d,
    nlist,
    m,
    bits
)

index.train(xb)

index.add(xb)

index.nprobe = 10

D, I = index.search(xq, k)

print(I)
print(D)




# Vectors
#       ↓
# Train IVF
#       ↓
# Create clusters
#       ↓
# Train PQ
#       ↓
# Compress vectors
#       ↓
# Store compressed vectors
# inside clusters
#       ↓
# Search nearest clusters
#       ↓
# Approximate distance