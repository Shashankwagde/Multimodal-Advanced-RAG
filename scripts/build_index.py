import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.indexing import build_index
from src.ingestion import load_documents_from_folder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main() -> None:
    """
    Load the documents folder, split it and build the index from scratch.

    Prints a short summary so you can see how many chunks were created. Any
    failure (missing API key, empty folder) is reported as a readable message
    rather than a stack trace.
    """    
    print(f"Reading documents from: {config.DOCUMENTS_DIR}")

    documents = load_documents_from_folder(config.DOCUMENTS_DIR)

    if not documents:
        print("No readable documents found. Add PDF, TXT or DOCX files and try again.")
        return
    
    try:
        stats = build_index(documents)
    except Exception as e:
        print(f"Build failed: {e}")        
        return
    
    print("\nIndex built successfully.")
    print(f"    Pages / documents loaded : {stats['documents']}")
    print(f"    Parent Chunks(Context) : {stats['parents']}")
    print(f"    Child Chunks(searched) : {stats['children']}")
    print(f"    Vector database        : {config.VECTOR_DB}")
    print("\nNow run: streamlit run app.py")

if __name__=="__main__":
    main()