
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# import faiss
import pandas as pd


def tokenize_text(text):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
    )
    texts = []
    for page in text:
        page_texts = text_splitter.split_text(page.page_content)
        texts.extend(page_texts)
    
    return texts

def load_pdf_with_langchain(pdf_path):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    return document

pdf= load_pdf_with_langchain('Data/2023 EU-wide stress test - Methodological Note.pdf')

texts= tokenize_text(pdf)

# model_embeddings = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-mpnet-base-v2")

# embedd= model_embeddings.embed_documents(texts)

# e_df= pd.DataFrame(embedd)
# e_df.to_csv('test.csv')

