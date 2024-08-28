from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import faiss


class TextProcessors:


    def __init__(self, path, embeddings_model):
        self.path = path
        self.embeddings_model = embeddings_model
        self.embeddings = None
        self.index = None
        self.texts = None

    def tokenize_text(self, text):
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
        )
        texts = []
        for page in text:
            page_texts = text_splitter.split_text(page.page_content)
            texts.extend(page_texts)
        
        return texts
    
    def pdf_file (self):

        def load_pdf_with_langchain(pdf_path):
            loader = PyPDFLoader(pdf_path)
            document = loader.load()
            return document
        
        pdf= load_pdf_with_langchain (self.path)
        self.texts= self.tokenize_text(pdf)
        model_embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
        self.embeddings = model_embeddings.embed_documents(self.texts)
        self.index = FAISS.from_documents(self.texts, model_embeddings)

    def query_document(self, query, k=5):
        
        model_embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
        query_embedding = model_embeddings.embed_query(query)
        results = self.index.similarity_search_by_vector(query_embedding, k=k)
        return results


        
        

    


