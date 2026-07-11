import faiss
import numpy as np
import time


def benchmark_indexes(dimension=128, num_vectors=100000, k=5):
   
    # Step 1: Generate sample dataset
    vectors = np.random.random((num_vectors, dimension)).astype("float32")
    query = np.random.random((1, dimension)).astype("float32")

    # Step 2: Create the indexes

    # Flat Index (Exact Search)
    flat_index = faiss.IndexFlatL2(dimension)

    # IVF Index
    nlist = int(np.sqrt(num_vectors))   # Number of clusters
    quantizer = faiss.IndexFlatL2(dimension)
    ivf_index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        nlist
    )

    # HNSW Index
    hnsw_index = faiss.IndexHNSWFlat(
        dimension,
        16      # M = Number of neighbors
    )


    # Step 3: Build each index

    # Flat
    flat_index.add(vectors)

    # IVF
    ivf_index.train(vectors)
    ivf_index.add(vectors)
    ivf_index.nprobe = 10

    # HNSW
    hnsw_index.hnsw.efConstruction = 40
    hnsw_index.hnsw.efSearch = 16
    hnsw_index.add(vectors)


    # Step 4: Benchmark Search Time
    results = {}

    indexes = [
        ("Flat", flat_index),
        ("IVF", ivf_index),
        ("HNSW", hnsw_index)
    ]

    for name, index in indexes:

        start = time.time()
        # Run the same query 100 times
        for _ in range(100):
            index.search(query, k)

        avg_time = (time.time() - start) / 100
        results[name] = avg_time

    return results


# Run Benchmark
results = benchmark_indexes()
print("Average Search Time (seconds)\n")

for index_name, avg_time in results.items():
    print(f"{index_name:6} : {avg_time:.8f} seconds/query")