# Data ingestion in Pinecone means taking your raw data, converting it into vectors (embeddings),
# and storing it in the index.

# It’s basically the process of uploading and indexing your data so it can be searched later.

######################################################
# 1. Update Records
######################################################

# This is about Add or update records in Pinecone indexes and manage data with namespaces.

# -----------------------------------------------------
# 1.1.1 Upsert dense vectors (text)
# -----------------------------------------------------

import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_HOST = os.getenv("INDEX_HOST")

pc = Pinecone(API_KEY)

index = pc.Index(INDEX_HOST)

# Upsert records into a namespace
# `chunk_text` fields are converted to dense vectors
# `category` fields are stored as metadata

index.upsert_records(
    "example-namespace",
    [
        {
            "_id": "rec1",
            "chunk_text": "Apples are a great source of dietary fiber, which supports digestion and helps maintain a healthy gut.",
            "category": "digestive system",
        },
{
            "_id": "rec2",
            "chunk_text": "Apples originated in Central Asia and have been cultivated for thousands of years, with over 7,500 varieties available today.",
            "category": "cultivation",
        },
        {
            "_id": "rec3",
            "chunk_text": "Rich in vitamin C and other antioxidants, apples contribute to immune health and may reduce the risk of chronic diseases.",
            "category": "immune system",
        },
        {
            "_id": "rec4",
            "chunk_text": "The high fiber content in apples can also help regulate blood sugar levels, making them a favorable snack for people with diabetes.",
            "category": "endocrine system",
        },
    ]
)

# Wait for the upserted vectors to be indexed
import time
time.sleep(10)

# -----------------------------------------------------
# 1.1.2 Upsert dense vectors (vectors)
# -----------------------------------------------------

from pinecone.grpc import PineconeGRPC as Pinecone

pc = Pinecone(API_KEY)

index = pc.Index(INDEX_HOST)

index.upsert(
    vector = [
        {
            "id": "A",
            "values": [0.1, 0.2, 0.3, 0.4, 0.5],
            "metadata": {"genre": "comedy", "year": 2020}
        },
        {
            "id": "B",
            "values": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "metadata": {"genre": "documentary", "year": 2019}
        },
        {
            "id": "C",
            "values": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            "metadata": {"genre": "comedy", "year": 2019}
        },
        {
            "id": "D",
            "values": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
            "metadata": {"genre": "drama"}
        }
    ],
    namsapce = "example-namespace"
)

# -----------------------------------------------------
# 1.2.1 Upsert sparse vectors (text)
# -----------------------------------------------------

from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# To get the unique host for an index,
# see https://docs.pinecone.io/guides/manage-data/target-an-index
index = pc.Index(host="INDEX_HOST")

# Upsert records into a namespace
# `chunk_text` fields are converted to sparse vectors
# `category` and `quarter` fields are stored as metadata
index.upsert_records(
    "example-namespace",
    [
        {
            "_id": "vec1",
            "chunk_text": "AAPL reported a year-over-year revenue increase, expecting stronger Q3 demand for its flagship phones.",
            "category": "technology",
            "quarter": "Q3"
        },
        {
            "_id": "vec2",
            "chunk_text": "Analysts suggest that AAPL'\''s upcoming Q4 product launch event might solidify its position in the premium smartphone market.",
            "category": "technology",
            "quarter": "Q4"
        },
        {
            "_id": "vec3",
            "chunk_text": "AAPL'\''s strategic Q3 partnerships with semiconductor suppliers could mitigate component risks and stabilize iPhone production.",
            "category": "technology",
            "quarter": "Q3"
        },
        {
            "_id": "vec4",
            "chunk_text": "AAPL may consider healthcare integrations in Q4 to compete with tech rivals entering the consumer wellness space.",
            "category": "technology",
            "quarter": "Q4"
        }
    ]
)

time.sleep(10) # Wait for the upserted vectors to be indexed


# -----------------------------------------------------
# 1.2.2 Upsert sparse vectors (vectors)
# -----------------------------------------------------

from pinecone import Pinecone, SparseValues, Vector

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index(host="INDEX_HOST")

index.upsert(
    namespace="example-namespace",
    vectors=[
        {
            "id": "vec1",
            "sparse_values": {
                "values": [1.7958984, 0.41577148, 2.828125, 2.8027344, 2.8691406, 1.6533203, 5.3671875, 1.3046875, 0.49780273, 0.5722656, 2.71875, 3.0820312, 2.5019531, 4.4414062, 3.3554688],
                "indices": [822745112, 1009084850, 1221765879, 1408993854, 1504846510, 1596856843, 1640781426, 1656251611, 1807131503, 2543655733, 2902766088, 2909307736, 3246437992, 3517203014, 3590924191]
            },
            "metadata": {
                "chunk_text": "AAPL reported a year-over-year revenue increase, expecting stronger Q3 demand for its flagship phones.",
                "category": "technology",
                "quarter": "Q3"
            }
        },
        {
            "id": "vec2",
            "sparse_values": {
                "values": [0.4362793, 3.3457031, 2.7714844, 3.0273438, 3.3164062, 5.6015625, 2.4863281, 0.38134766, 1.25, 2.9609375, 0.34179688, 1.4306641, 0.34375, 3.3613281, 1.4404297, 2.2558594, 2.2597656, 4.8710938, 0.5605469],
                "indices": [131900689, 592326839, 710158994, 838729363, 1304885087, 1640781426, 1690623792, 1807131503, 2066971792, 2428553208, 2548600401, 2577534050, 3162218338, 3319279674, 3343062801, 3476647774, 3485013322, 3517203014, 4283091697]
            },
            "metadata": {
                "chunk_text": "Analysts suggest that AAPL'\''s upcoming Q4 product launch event might solidify its position in the premium smartphone market.",
                "category": "technology",
                "quarter": "Q4"
            }
        },
        {
            "id": "vec3",
            "sparse_values": {
                "values": [2.6875, 4.2929688, 3.609375, 3.0722656, 2.1152344, 5.78125, 3.7460938, 3.7363281, 1.2695312, 3.4824219, 0.7207031, 0.0826416, 4.671875, 3.7011719, 2.796875, 0.61621094],
                "indices": [8661920, 350356213, 391213188, 554637446, 1024951234, 1640781426, 1780689102, 1799010313, 2194093370, 2632344667, 2641553256, 2779594451, 3517203014, 3543799498, 3837503950, 4283091697]
            },
            "metadata": {
                "chunk_text": "AAPL'\''s strategic Q3 partnerships with semiconductor suppliers could mitigate component risks and stabilize iPhone production",
                "category": "technology",
                "quarter": "Q3"
            }
        },
        {
            "id": "vec4",
            "sparse_values": {
                "values": [0.73046875, 0.46972656, 2.84375, 5.2265625, 3.3242188, 1.9863281, 0.9511719, 0.5019531, 4.4257812, 3.4277344, 0.41308594, 4.3242188, 2.4179688, 3.1757812, 1.0224609, 2.0585938, 2.5859375],
                "indices": [131900689, 152217691, 441495248, 1640781426, 1851149807, 2263326288, 2502307765, 2641553256, 2684780967, 2966813704, 3162218338, 3283104238, 3488055477, 3530642888, 3888762515, 4152503047, 4177290673]
            },
            "metadata": {
                "chunk_text": "AAPL may consider healthcare integrations in Q4 to compete with tech rivals entering the consumer wellness space.",
                "category": "technology",
                "quarter": "Q4"
            }
        }
    ]
)

# -----------------------------------------------------
# 1.3 Upsert records in batches
# -----------------------------------------------------

import random
import itertools
from pinecone.grpc import PineconeGRPC as Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# Connect to your Pinecone index using its host
index = pc.Index(host="INDEX_HOST")


def chunks(iterable, batch_size=200):
    """
    Break a large iterable into smaller chunks (batches).
    Uses iter + islice + yield for memory-efficient processing.
    """

    # Convert iterable (like map generator) into an iterator
    # Iterator keeps track of "where we are" in the data
    it = iter(iterable)

    # Take the first 'batch_size' items from iterator
    # islice pulls items one-by-one WITHOUT loading everything
    chunk = tuple(itertools.islice(it, batch_size))

    # Loop until no more data is left
    while chunk:
        # yield returns the chunk and PAUSES the function here
        # Next time it's called, it resumes from below
        yield chunk

        # Get the NEXT batch from where iterator left off
        # This does NOT restart — it continues from previous position
        chunk = tuple(itertools.islice(it, batch_size))


# Each vector will have 128 dimensions
vector_dim = 128

# Total number of vectors to generate
vector_count = 10000


# map() creates a LAZY generator (not stored in memory)
# Each item looks like: ("id-1", [0.12, 0.45, ...])
example_data_generator = map(
    lambda i: (f'id-{i}', [random.random() for _ in range(vector_dim)]),
    range(vector_count)
)


# Process data in batches of 200
for ids_vectors_chunk in chunks(example_data_generator, batch_size=200):

    # Each iteration:
    # - chunks() resumes from last yield
    # - returns next 200 items
    # - does NOT recompute or restart

    index.upsert(vectors=ids_vectors_chunk)

