#%%
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama

def vectorstore(doc_splits):
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name= f'collection',
        embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text'),
        )

    return vectorstore

def search (vectorstore_db,question):
    return vectorstore_db.similarity_search_with_score(question, k=5)

def metadata (answer):
    metadata= []
    for i in answer:
        meta= i[0].metadata
        if meta not in metadata:
            metadata.append(meta)
    
    pages= str([int(i['page'])+1 for i in metadata]).strip('[]')

    def extract_filename(source):
        index_slash = source.rfind('/')
        index_backslash = source.rfind("\\")
        index = max(index_slash, index_backslash)
        return source[index+1:]
    
    documents= str(set([extract_filename(i['source']) for i in metadata])).strip('{}')

    return [documents, pages]
    
    



# %%
