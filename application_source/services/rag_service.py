import faiss
import pickle

from FlagEmbedding import FlagModel

from services.utilities import SingletonMeta

class PolicyRag(metaclass=SingletonMeta):
    
    def __init__(self):

        with open("data/input/policy_embeddings.pkl", "rb") as f:
            policy_embeddings = pickle.load(f)

        self.docs = [doc.page_content for doc in policy_embeddings["docs"]]
        embeddings = policy_embeddings["embeddings"]
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        self.model = FlagModel('BAAI/bge-small-en-v1.5', use_fp16=True)

    def retrieve(self, query, top_k=10):

        query_vector = self.model.encode([query])

        distances, indices = self.index.search(query_vector, top_k)
        retrieved_chunks = [self.docs[i] for i in indices[0] if i < len(self.docs)]

        print(f"Retrieved {len(retrieved_chunks)} chunk from vector store")

        context = f"Here are policy documents:\n\n"
        context += "\n------------------\n".join(retrieved_chunks) 

        return context
    
if __name__ == "__main__":

    rag = PolicyRag()
    res = rag.retrieve(query="What software is approved by the company?")
    print(res)