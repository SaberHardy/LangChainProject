import os
import sys

from dotenv import load_dotenv


def setup_environment():
    """
    Setup the environment for the RAG system.
    This function loads environment variables from a .env file and sets up necessary paths.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Add src directory to sys.path for module imports
    directories = ["./data/documents",
                   "./storage",
                   "./Secrets"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Ensured directory exists: {directory}")

    documents_dir = "./data/documents"
    if os.path.exists(documents_dir):
        print(f"✅ Documents directory is set up at: {documents_dir}")
        files = [f for f in os.listdir(documents_dir) if os.path.isfile(os.path.join(documents_dir, f))]
        if files:
            print(f"- Found {len(files)} files in documents directory.")
        else:
            print("- No files found in documents directory. Please add documents for processing.")

    try:
        from src.config import Config

        Config.validate_config(Config)
        Config.print_config()
    except Exception as e:
        print(f"❌ Error during configuration validation: {e}")
        sys.exit(1)

    print("✅ Environment setup complete.")

if __name__ == "__main__":
    setup_environment()