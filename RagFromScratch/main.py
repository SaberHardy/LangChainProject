import os
import sys
from dotenv import load_dotenv

from RagFromScratch.src.rag_chain import RAGSystemChain

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.document_processor import DocumentProcessor
from src.vector_store_local import VectorStoreManager
from src.config import Config


class RAGFromScratchApp:
    """
    Main application class for the RAG From Scratch system.
    This class initializes the environment, processes documents, sets up the vector store,
    and creates the RAG chain for answering user queries.
    """

    def __init__(self, data_folder=None):
        self.data_folder = data_folder or Config.DATA_FOLDER
        self.processor = DocumentProcessor()
        self.vs_manager = None
        self.rag_system = None
        self.chain = None

    def initialize_environment(self, rebuild_vector_store=False):
        """
        Initialize the environment by loading configurations and validating them.
        """
        try:
            Config.validate_config(Config)
            print("✅ Environment initialized and configuration validated.")
            Config.print_config()
            self.vs_manager = VectorStoreManager()

            vector_store_exists = os.path.exists(Config.PERSIST_DIRECTORY)
            if rebuild_vector_store or not vector_store_exists:
                print("🔄 Building or rebuilding the vector store...")
                documents = self.processor.load_documents(folder_path=self.data_folder)

                if not documents:
                    print("❌ No documents found to process. Please add documents to the data folder.")
                    print("Supported file formats are: .txt, .pdf, .docx")
                    return False

                chunks = self.processor.chunk_documents(documents)
                self.vs_manager.create_vector_store(chunks)
            else:
                print("🔍 Loading existing vector store...")
                doc_count = self.vs_manager.get_doc_count()
                print(f"✅ Loaded vector store with {doc_count} documents.")

            retriever = self.vs_manager.get_retriever()
            self.rag_system = RAGSystemChain()
            self.chain = self.rag_system.create_rag_chain(retriever)

            print("\n✅ RAG System Ready!")
            print("   - Local embeddings: ✅ (no API limits)")
            print("   - Gemini chat: ✅")
            print("   - Document retrieval: ✅")

            return True

        except Exception as e:
            print(f"❌ Error during environment initialization: {e}")
            return False

    def query(self, question):
        """
        Query the RAG chain with a question.
        :param question: The question to ask.
        :return: The generated answer.
        """
        if not self.chain:
            print("❌ RAG chain is not initialized. Please initialize the environment first.")
            return None

        try:
            print(f"❓ Querying RAG Chain with question: {question}")
            return self.rag_system.query(self.chain, question)

        except Exception as e:
            print(f"❌ Error during query: {e}")
            return None

    def interactive_mode(self):
        """
        Start an interactive mode for user queries.
        """
        print("\n" + "=" * 50)
        print("🤖 RAG System - Interactive Mode")
        print("=" * 50)
        print("Type your questions below (type 'quit', 'exit', or 'q' to stop)")
        print("Type 'debug' to see retrieval details")
        print("-" * 50)

        debug_mode = False

        while True:
            try:
                question = input("\n💬 Your Question: "+ "\n\n").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif question.lower() == 'debug':
                    debug_mode = not debug_mode
                    status = "ON" if debug_mode else "OFF"
                    print(f"🔧 Debug mode {status}")
                    continue
                elif question.lower() == '':
                    continue

                if debug_mode:
                    print("🔍 Retrieving documents...")
                    retriever = self.vs_manager.get_retriever()
                    docs = retriever.get_relevant_documents(question)
                    print(f"📄 Retrieved {len(docs)} documents:")
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get('source', 'Unknown')
                        print(f"   {i}. {source} ({len(doc.page_content)} chars)")
                    print("-" * 30)

                # Get answer
                answer = self.query(question)
                print(f"\n🤖 Answer: {answer}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function"""
    print("🎯 RAG System with Google Gemini")
    print("   Built with LangChain + Local Embeddings")

    # Initialize application
    app = RAGFromScratchApp()

    # Check if this is first run or rebuild requested
    first_run = not os.path.exists(Config.PERSIST_DIRECTORY)
    rebuild = first_run or (len(sys.argv) > 1 and sys.argv[1] == "--rebuild")

    if rebuild and not first_run:
        print("♻️ Rebuilding vector store...")

    # Initialize system
    if app.initialize_environment(rebuild_vector_store=rebuild):
        # Start interactive mode
        app.interactive_mode()
    else:
        print("❌ Failed to initialize RAG system")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check your Google AI API key in Secrets/gcp_keys.py")
        print("   2. Add documents to data/documents/ folder")
        print("   3. Run: python setup.py to verify setup")


if __name__ == "__main__":
    main()
