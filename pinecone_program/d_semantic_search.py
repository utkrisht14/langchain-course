from c_upsert_text import dense_index

# Define the query
query = "Famous historical structures and monuments"

# Search the dense index
results = dense_index.search(
    namespace="example-namespace",
    query = {
        "top_k": 10,
        "inputs": {
            "text": query
        }
    }
)

# Print the results
for hit in results["result"]["hits"]:
    print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}")