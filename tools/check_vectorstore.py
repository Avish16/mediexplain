import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VECTOR_STORE_ID = "vs_6930ffbfc0188191997f62a2ebe5daf5"   # your ID

# Retrieve vector store info
vs = client.vector_stores.retrieve(VECTOR_STORE_ID)
print("\n=== VECTOR STORE INFO ===")
print(vs)

print("\n=== FILES IN VECTOR STORE ===")
files = client.vector_stores.files.list(vector_store_id=VECTOR_STORE_ID)

for f in files.data:
    # Fetch the real File object from the Files API
    file_obj = client.files.retrieve(f.id)

    print(
        f"- File ID: {f.id} | "
        f"Filename: {file_obj.filename} | "
        f"Bytes: {file_obj.bytes} | "
        f"Status: {f.status}"
    )
