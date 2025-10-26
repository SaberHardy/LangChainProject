import os
from dotenv import load_dotenv
from Secrets.openai_key import google_api_key

load_dotenv()


class Config:
    """
    Configuration class for the RAG system.
    This class loads environment variables and provides access to configuration settings.
    It includes settings for the OpenAI API key, the directory for storing vector data,
    and any other relevant configuration parameters needed for the RAG system to function properly.
    """
    GOOGLE_API_KEY = google_api_key
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v"
    CHAT_MODEL = "gemini-1.5-flash"

    CHUNK_SIZE = 1000  # Size of text chunks for processing
    CHUNK_OVERLAP = 200  # Overlap between chunks to maintain context

    DATA_FOLDER = os.getenv("DATA_FOLDER", "./data/documents")
    PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./storage/chroma_db")

    SEARCH_K = 4  # Number of top results to retrieve from the vector store

    @staticmethod
    def validate_config(cls):
        """
        Validate that all required configurations are present
        :return:
        """
        if not cls.GOOGLE_API_KEY:
            raise ValueError("Google API key is missing. Please set the GOOGLE_API_KEY environment variable.")

        return True

    @classmethod
    def load_config(cls):
        """
        Load and validate the configuration settings.
        :return:
        """
        print("🔧 RAG System Configuration:")
        print(f"   - Chat Model: {cls.CHAT_MODEL}")
        print(f"   - Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"   - Data Folder: {cls.DATA_FOLDER}")
        print(f"   - Chunk Size: {cls.CHUNK_SIZE}")
        print(f"   - Documents to Retrieve: {cls.SEARCH_K}")


if __name__ == "__main__":
    # Load and validate the configuration when this script is run directly
    if Config.validate_config(Config):
        Config.load_config()
        print("Configuration loaded successfully. All required settings are present.")
    else:
        print("Configuration validation failed. Please check the required settings and try again.")
