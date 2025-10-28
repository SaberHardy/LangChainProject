import os

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma
from Secrets.openai_key import google_api_key

GOOGLE_API_KEY = google_api_key


class VectorStoreManager:
    def __init__(self, persist_directory=None):
        from RagFromScratch.src.config import Config

        self.persist_directory = persist_directory or Config.PERSIST_DIRECTORY

        # Create storage directory if it doesn't exist
        os.makedirs(os.path.dirname(self.persist_directory), exist_ok=True)
        print(f"Initializing VectorStoreManager with persist directory: {self.persist_directory}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def create_vector_store(self, documents):
        """
        Create a vector store from the given documents. 
        This method will generate embeddings for the documents and store them in a Chroma vector store.
        :param documents: 
        :return: 
        """
        try:
            print(f"🏗️ Creating vector store with {len(documents)} chunks...")

            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            # vector_store.persist()

            print(f"✅ Vector store created successfully!")
            print(f"   - Location: {self.persist_directory}")
            print(f"   - Documents: {len(documents)}")

            return vector_store
        except Exception as e:
            print(f"❌ An error occurred while creating the vector store: {e}")
            raise

    def load_vector_store(self):
        """
        Load an existing vector store from the persist directory. 
        This method will return a Chroma vector store if it exists, otherwise it will raise an error.
        :return: 
        """""
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"Vector store not found at {self.persist_directory}. Please create it first.")

        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
        print(f"Vector store loaded from {self.persist_directory}.")
        return vector_store

    def get_retriever(self, search_type="similarity", k=4):
        """
        Get a retriever from the given vector store. 
        This method will return a retriever that can be used to query the vector store for relevant documents.
        :param search_type: 
        :param k: 
        :return: 
        """
        from RagFromScratch.src.config import Config

        k = k or Config.SEARCH_K  # Use default from config if k is not provided
        print(f"Creating retriever with search type '{search_type}' and k={k}...")

        vector_store = self.load_vector_store()

        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

        print("Retriever created from the vector store with search type '{}' and k={}".format(search_type, k))
        return retriever

    def get_doc_count(self):
        try:
            vector_store = self.load_vector_store()
            doc_count = vector_store._collection.count()
            print(f"Vector store contains {doc_count} documents.")
            return doc_count
        except FileNotFoundError:
            print(f"Vector store not found at {self.persist_directory}. Please create it first.")
            return 0
        except Exception as e:
            print(f"An error occurred while counting documents in the vector store: {e}")
            return 0


if __name__ == "__main__":
    from document_processor import DocumentProcessor
    from Secrets.openai_key import google_api_key

    processor = DocumentProcessor()
    docs = processor.load_documents("../data/documents")
    chunks = processor.chunk_documents(docs)

    # Pass the actual API key
    vs_manager = VectorStoreManager()
    vector_store = vs_manager.create_vector_store(chunks)

    retriever = vs_manager.get_retriever()
    test_results = retriever.invoke("What is the main topic of the documents?")

    print(f"Retrieved {len(test_results)} relevant documents:")

    print("vector store document count:", vs_manager.get_doc_count())
    print(f"Vector Store: {vector_store}")
