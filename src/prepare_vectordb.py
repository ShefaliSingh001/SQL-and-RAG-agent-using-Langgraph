import os
import yaml
from pyprojroot import here
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"


class PrepareVectorDB:
    """
    A class to prepare and manage a Vector Database (VectorDB) using 
    HuggingFace local embeddings.
    """

    def __init__(self,
                 doc_dir: str,
                 chunk_size: int,
                 chunk_overlap: int,
                 embedding_model: str,
                 vectordb_dir: str,
                 collection_name: str
                 ) -> None:

        self.doc_dir = doc_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.collection_name = collection_name
        # Initialize the Hugging Face Embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def path_maker(self, file_name: str, doc_dir: str) -> str:
        return os.path.join(str(here(doc_dir)), file_name)

    def run(self) -> None:
        if not os.path.exists(here(self.vectordb_dir)):
            file_list = [
                fn for fn in os.listdir(here(self.doc_dir)) 
                if fn.lower().endswith('.pdf')
            ]
            
            if not file_list:
                print(f"No PDF documents found in '{self.doc_dir}'. Skipping DB creation.")
                return

            os.makedirs(here(self.vectordb_dir), exist_ok=True)
            print(f"Directory '{self.vectordb_dir}' was created.")

            docs = [PyPDFLoader(self.path_maker(
                fn, self.doc_dir)).load_and_split() for fn in file_list]
            docs_list = [item for sublist in docs for item in sublist]

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            doc_splits = text_splitter.split_documents(docs_list)

            # Create Vector DB with HuggingFace Embeddings
            vectordb = Chroma.from_documents(
                documents=doc_splits,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                persist_directory=str(here(self.vectordb_dir))
            )
            print("VectorDB created and saved successfully.")
            print("Number of vectors in vectordb:",
                  vectordb._collection.count(), "\n")
        else:
            print(f"Directory '{self.vectordb_dir}' already exists.")


if __name__ == "__main__":
    load_dotenv()

    with open(here("configs/tools_config.yml")) as cfg:
        app_config = yaml.load(cfg, Loader=yaml.FullLoader)

    # Configs for swiss airline policy document
    chunk_size = app_config["swiss_airline_policy_rag"]["chunk_size"]
    chunk_overlap = app_config["swiss_airline_policy_rag"]["chunk_overlap"]
    embedding_model = app_config["swiss_airline_policy_rag"]["embedding_model"]
    vectordb_dir = app_config["swiss_airline_policy_rag"]["vectordb"]
    collection_name = app_config["swiss_airline_policy_rag"]["collection_name"]
    doc_dir = app_config["swiss_airline_policy_rag"]["unstructured_docs"]

    prepare_db_instance = PrepareVectorDB(
        doc_dir=doc_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        vectordb_dir=vectordb_dir,
        collection_name=collection_name)

    prepare_db_instance.run()

    # Configs for stories document
    chunk_size = app_config["stories_rag"]["chunk_size"]
    chunk_overlap = app_config["stories_rag"]["chunk_overlap"]
    embedding_model = app_config["stories_rag"]["embedding_model"]
    vectordb_dir = app_config["stories_rag"]["vectordb"]
    collection_name = app_config["stories_rag"]["collection_name"]
    doc_dir = app_config["stories_rag"]["unstructured_docs"]

    prepare_db_instance = PrepareVectorDB(
        doc_dir=doc_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        vectordb_dir=vectordb_dir,
        collection_name=collection_name)

    prepare_db_instance.run()