#%%
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from tokenizer import tokens
from vectorstore import vectorstore, search, metadata

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}



---

Answer the question based on the above context: {question}
"""
#%%
def process_input(pdfs,question):
    model_local = ChatOllama(model="llama3.1", temperature= 0)
    doc_splits= tokens(pdfs)
    vectorstore_db= vectorstore(doc_splits)
    vectorstore_results= search(vectorstore_db, question)
    meta_= metadata(vectorstore_results)
    # page= retriever.invoke(question)[0].metadata
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in vectorstore_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question, sources= meta_[0], pages= meta_[1])
    response_text = model_local.invoke(prompt)
    # final_answer= StrOutputParser(response_text)

    text= response_text.content + f'\nPages: {meta_[1]} | Source: {meta_[0]}'
    
    return text


#%%

iface = gr.Interface(fn=process_input,
                     inputs=[gr.Textbox(label="Enter PDF path"),
                             gr.Textbox(label="Question")],
                     outputs="text",
                     title="Document Query",
                     description="Enter PDFs and a question to query the documents.")
iface.launch()



