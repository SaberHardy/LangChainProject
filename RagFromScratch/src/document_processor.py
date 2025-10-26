from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os


class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                            chunk_overlap=chunk_overlap,
                                                            length_function=len)

    def load_documents(self, folder_path):
        documents = []

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if file_name.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_name.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            elif file_name.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            else:
                print(f"Unsupported file format: {file_name}. Skipping.")
                continue

            print(f"Loaded {len(documents)} documents from {file_name}.")
        return documents

    def chunk_documents(self, documents):
        chunks = self.text_splitter.split_documents(documents)
        print("Chunked documents into {} chunks.".format(len(chunks)))

        return chunks

# if __name__ == "__main__":
#     processor = DocumentProcessor()
#     docs = processor.load_documents("./data")
#     chunks = processor.chunk_documents(docs)