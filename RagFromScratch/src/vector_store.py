import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from Secrets.openai_key import google_api_key

GOOGLE_API_KEY = google_api_key


class VectorStoreManager:
    def __init__(self, persist_directory=".storage/chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY)

    def create_vector_store(self, documents):
        """
        Create a vector store from the given documents. 
        This method will generate embeddings for the documents and store them in a Chroma vector store.
        :param documents: 
        :return: 
        """""
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        vector_store.persist()

        print(f"Vector store created and persisted at {self.persist_directory}.")
        return vector_store

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
        :return: 
        """""
        vector_store = self.load_vector_store()
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        print("Retriever created from the vector store with search type '{}' and k={}".format(search_type, k))
        print("Retriever created from the vector store.")
        return retriever


if __name__ == "__main__":
    from document_processor import DocumentProcessor
    from Secrets.openai_key import google_api_key

    processor = DocumentProcessor()
    docs = processor.load_documents("../data")
    chunks = processor.chunk_documents(docs)

    # Pass the actual API key
    vs_manager = VectorStoreManager()
    vs_manager.embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    vector_store = vs_manager.create_vector_store(chunks)
