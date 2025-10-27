from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from RagFromScratch.src.vector_store_local import VectorStoreManager


class RAGSystemChain:
    """
    RAG System Chain class that sets up the retrieval-augmented generation (RAG) process.
    This class initializes the chat model, prompt template, and output parser for generating responses
    based on retrieved documents.
    """

    def __init__(self):
        from RagFromScratch.src.config import Config

        Config.validate_config(Config)
        print(f"🔧 Initializing RAG System Chain...: {Config.CHAT_MODEL}")
        self.llm = ChatGoogleGenerativeAI(model=Config.CHAT_MODEL,
                                          google_api_key=Config.GOOGLE_API_KEY,
                                          temperature=0.1,
                                          max_retries=2)
        self.setup_prompt_template()
        print("✅ RAG System Chain initialized successfully.")

    def setup_prompt_template(self):
        """
        Set up the chat prompt template and output parser for the RAG system.
        :return:
        """

        self.output_parser = StrOutputParser()
        self.prompt_template = ChatPromptTemplate.from_template("""You are a helpful and accurate AI assistant. Use the following context to answer the user's question.
            CONTEXT:
            {context}
            
            QUESTION:
            {question}

            INSTRUCTIONS:
            - Answer the question based ONLY on the provided context
            - If the context doesn't contain the answer, say "I don't have enough information to answer this question based on the provided documents."
            - Do not make up information or use outside knowledge
            - Keep your answer concise and relevant to the question
            - If the question is unclear, ask for clarification
            
            ANSWER:
            """)

    def create_rag_chain(self, retriever):
        """
        Create the RAG chain using the retriever and the initialized components.
        :param retriever: The document retriever to use for fetching relevant documents.
        :return: The configured RAG chain.
        """
        from langchain.chains import RetrievalQA

        def format_documents(documents):
            """
            Format the retrieved documents into a single context string.
            :param documents: List of retrieved documents.
            :return: Formatted context string.
            """
            if not documents:
                return "No relevant documents found."

            formatted_docs = []
            for i, doc in enumerate(documents, 1):
                source = doc.metadata.get("source", "Unknown Source")
                content = doc.page_content.strip()
                formatted_docs.append(f"Document {i} (Source: {source}):\n{content}\n")

            return "\n\n".join(formatted_docs)

        rag_chain = (
                {
                    "context": retriever | format_documents,
                    "question": RunnablePassthrough(),
                }
                | self.prompt_template
                | self.llm
                | self.output_parser
        )
        return rag_chain

    def query(self, chain, question):
        """
        Query the RAG chain with a question.
        :param chain: The RAG chain to query.
        :param question: The question to ask.
        :return: The generated answer.
        """
        try:
            print(f"❓ Querying RAG Chain with question: {question}")
            return chain.invoke(question)

        except Exception as e:
            print(f"Error printing question: {e}")


if __name__ == "__main__":
    print("Testing the rag System chain...")

    try:
        rag_system = RAGSystemChain()
        vs_manager = VectorStoreManager()

        # Get Retriever
        retriever = vs_manager.get_retriever()

        # Create RAG Chain
        rag_chain = rag_system.create_rag_chain(retriever)

        # Test Query
        test_question = "What is this document about?"
        print("The question is: ", test_question)
        answer = rag_system.query(rag_chain, test_question)
        print("The answer is:", answer)

    except Exception as e:
        print(f"Error during RAG System Chain testing: {e}")
