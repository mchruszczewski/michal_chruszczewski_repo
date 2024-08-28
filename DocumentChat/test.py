#%%
from TextProcessors import TextProcessors
# %%
test_pdf= TextProcessors ('Data/2023 EU-wide stress test - Methodological Note.pdf', "sentence-transformers/all-mpnet-base-v2")
# %%
test_pdf.pdf_file()
# %%
query = "What is the main objective of the stress test?"
results = test_pdf.query_document(query, k=3)