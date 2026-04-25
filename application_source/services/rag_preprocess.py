from FlagEmbedding import FlagModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle

model = FlagModel('BAAI/bge-small-en-v1.5', use_fp16=True)

with open("data/input/IT_support_policy.txt", "r") as f:
    policies = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.create_documents([policies])
docs_list = [doc.page_content for doc in docs]
embeddings = model.encode(docs_list)

policy_embeddings = {"docs":docs, "embeddings":embeddings}

with open("data/input/policy_embeddings.pkl", "wb") as f:
    pickle.dump(policy_embeddings, f)

