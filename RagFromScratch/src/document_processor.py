from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from RagFromScratch.src.config import Config
import os


class DocumentProcessor:
    """Handles loading and processing documents"""

    def __init__(self, chunk_size=None, chunk_overlap=None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        print(f"Initializing DocumentProcessor with chunk size: {self.chunk_size} "
              f"and chunk overlap: {self.chunk_overlap}")

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                            chunk_overlap=self.chunk_overlap,
                                                            length_function=len,
                                                            separators=["\n\n", "\n", " ", "", "\t", ".", ". ", ",",
                                                                        "!", "?", ";", ":", "-", "_", "(", ")", "[",
                                                                        "]", "{", "}", "\"", "'"])
        self.supported_extensions = {'.pdf': PyPDFLoader,
                                     '.txt': TextLoader,
                                     '.docx': Docx2txtLoader
                                     }

    def load_documents(self, folder_path):
        documents = []

        if not os.path.exists(folder_path):
            print("Error: The specified folder path '{}' does not exist. "
                  "Please check the path and try again.".format(folder_path))
            return documents

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in self.supported_extensions:
                try:
                    loader_class = self.supported_extensions[file_ext]
                    loader = loader_class(file_path)
                    loaded_docs = loader.load()

                    for doc in loaded_docs:
                        doc.metadata['source'] = file_name  # Add source metadata to each document

                    documents.extend(loaded_docs)
                    print(f"Loaded '{len(loaded_docs)}' documents from {file_name}.")
                except Exception as e:
                    print(f"Error loading {file_name}: {e}. Skipping this file.")
            else:
                print(f"Unsupported file format: {file_name}. "
                      f"Skipping this file. Supported formats are: {', '.join(self.supported_extensions.keys())}")

        print(f"📊 Total documents loaded: {len(documents)}")
        return documents

    def chunk_documents(self, documents):
        """Split the loaded documents into smaller pieces based on the specified chunk size and overlap for processing."""
        if not documents:
            print("❌ No documents to chunk")
            return []

        print(f"Starting to chunk {len(documents)} documents with chunk size {self.chunk_size} "
              f"and chunk overlap {self.chunk_overlap}. This may take a moment...")

        chunks = self.text_splitter.split_documents(documents)
        print("Chunked documents into {} chunks.".format(len(chunks)))

        avg_chunk_length = sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0
        print(f"Average chunk length: {avg_chunk_length:.2f} characters.")

        return chunks


if __name__ == "__main__":
    processor = DocumentProcessor()
    docs = processor.load_documents(Config.DATA_FOLDER)
    chunks_f = processor.chunk_documents(docs)

    print(chunks_f)
