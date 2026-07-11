import faiss
import numpy as np

d = 128
nb = 100000
nq = 5

xb = np.random.random((nb, d)).astype("float32")
xq = np.random.random((nq, d)).astype("float32")

index = faiss.IndexScalarQuantizer(
    d,
    faiss.ScalarQuantizer.QT_8bit
)

index.train(xb)

index.add(xb)

k = 5

D, I = index.search(xq, k)

print(I)
print(D)



# Vectors
#       ↓
# Learn scalar codebooks
#       ↓
# Compress every dimension
#       ↓
# Store compressed vectors
#       ↓
# Search