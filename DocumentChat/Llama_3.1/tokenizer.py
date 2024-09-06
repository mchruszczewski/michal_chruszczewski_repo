from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter


def tokens (pdfs): 
   pdfs_list = pdfs.split("\n")
   docs = [PyPDFLoader(pdf).load() for pdf in pdfs_list]
   docs_list = [item for sublist in docs for item in sublist]
    
   text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=10)
   doc_splits = text_splitter.split_documents(docs_list)

   return doc_splits
