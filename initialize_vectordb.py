from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import GooglePalmEmbeddings
from langchain.schema import Document
import os, time
import pandas as pd
from tqdm import tqdm


pinecone.init(
	api_key=os.getenv('PINECONE_CLIENT_API'),
	environment='gcp-starter'
)
embeddings = GooglePalmEmbeddings()
db = Pinecone.from_existing_index(index_name='wine-db', embedding=embeddings)

df = pd.read_csv('data/winemag-data-130k-v2.csv')
df = df.fillna('')
df = df.sample(100)


def create_document_from_tuple(t):
    return Document(
        page_content=t.description,
        metadata={
            'country': t.country,
            'province': t.province,
            'name': t.title,
            'variety':t.variety,
            'winery':t.winery
        }
    )

docs = [create_document_from_tuple(row) for row in tqdm(df.itertuples(index=False))]

print('Uploading to vector db')
s = time.perf_counter()
db.add_documents(docs)
elapsed = time.perf_counter() - s
print("\033[1m" + f"Upload executed in {elapsed:0.2f} seconds." + "\033[0m")
