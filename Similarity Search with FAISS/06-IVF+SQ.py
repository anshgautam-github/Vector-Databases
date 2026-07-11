import faiss
import numpy as np

d = 128
nb = 100000
nq = 5

xb = np.random.random((nb, d)).astype("float32")
xq = np.random.random((nq, d)).astype("float32")

nlist = 100

quantizer = faiss.IndexFlatL2(d)

index = faiss.IndexIVFScalarQuantizer(
    quantizer,
    d,
    nlist,
    faiss.ScalarQuantizer.QT_8bit
)

index.train(xb)

index.add(xb)

index.nprobe = 10

k = 5

D, I = index.search(xq, k)

print(I)
print(D)


# Vectors
#       ↓
# Create clusters
#       ↓
# Compress dimensions
#       ↓
# Store compressed vectors
# inside clusters
#       ↓
# Search