from fastapi import File, UploadFile, FastAPI, Depends
import llama_index
from llama_index import download_loader
from pathlib import Path
from enum import Enum
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
# from llama_index.vector_stores import ChromaVectorStore
# import chromadb
import os
import shutil
from llama_index.vector_stores.types import VectorStoreQuery
from llama_index.schema import NodeWithScore
from typing import Optional,List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.vector_stores import ChromaVectorStore
# import chromadb
import openai 
import uuid
from pydantic import BaseModel
import pinecone
from llama_index.vector_stores import PineconeVectorStore
import aiofiles


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
api_key = "a812b3f1-6a93-4899-939e-13d5ee0e132e"
pinecone.init(api_key=api_key, environment="gcp-starter")


app = FastAPI()

origins = [
    "http://localhost:3000",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


openai.api_key = "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"
global SYSTEM_PROMPT
SYSTEM_PROMPT = "Act as a teacher. Present the solution to every query in a informative manner."

class InputSchema(BaseModel):
    chunk_size: int
    text_splitter: str
    embedding_model:str

class InputSchemaTwo(BaseModel):
	embedding_model:str
	top_k:int 
	query_mode:str

class Splitter(Enum):
    sentence = llama_index.text_splitter.SentenceSplitter
    token = llama_index.text_splitter.TokenTextSplitter


def format_user_prompt(query:str,retrieved_nodes:List[str]) -> str:
    context_str = "\n\n".join([f'{idx+1}' + r.get_content() for idx,r in enumerate(retrieved_nodes)])
    BASE_PROMPT = f"""\
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query}
    Answer: \
    """
    return BASE_PROMPT


def gpt_helper(user:str,system:str = SYSTEM_PROMPT) -> str:
    model_id = "gpt-3.5-turbo-0613"
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ],
        max_tokens=3200,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]




@app.post("/upload")
async def upload_and_save_vectors(model_config:InputSchema = Depends(),upload_file:UploadFile = File(...)):
	# timer 1 start
	document_id = "document_"+str(uuid.uuid4())
	PDFReader = download_loader("PDFReader")
	config = model_config.dict()
	loader = PDFReader()
	destination_dir = os.getcwd() + "/storage"
	os.makedirs(destination_dir,exist_ok =True)
	destination = destination_dir + "/" +  document_id + ".pdf"
	async with aiofiles.open(destination, 'wb') as out_file:
		content = await upload_file.read()
		await out_file.write(content)
	# timer 1 end
	# timer 2 start
	documents = loader.load_data(destination) # take from payload
	print(len(documents))
	text_splitter = Splitter[config["text_splitter"]].value(chunk_size=config["chunk_size"])
	node_parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)
	nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)
	# timer 2 end
	# timer 3 start
	embedding_model = HuggingFaceEmbedding(model_name=config["embedding_model"])

#parallelize this
	for node in nodes:
		node_embedding = embedding_model.get_text_embedding(node.get_content())
		node.embedding = node_embedding
	# timer 3 end
	# timer 4 start
	try:
		pinecone.delete_index("quickstart")
	except Exception as e:
		pass
	pinecone.create_index("quickstart", dimension=768, metric="euclidean")
	pinecone_index = pinecone.Index("quickstart")
	data_store = PineconeVectorStore(pinecone_index=pinecone_index)
	data_store.add(nodes)
	# timer 4 end




	# db_path = os.getcwd() + "/chroma_db"
	# os.makedirs(db_path,exist_ok =True)
	# chroma_client = chromadb.PersistentClient(path=db_path)
	# chroma_collection = chroma_client.get_or_create_collection(document_id)
	# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
	# vector_store.add(nodes)
	return document_id



@app.get("/query")
def fetch_rag_response(query:str,system_prompt:str,document_id:str,model_config:InputSchemaTwo = Depends()) -> List[str]:
	config = model_config.dict()
	conversation_id = "conversation_"+str(uuid.uuid4())
	#timer 1 start
	embedding_model = HuggingFaceEmbedding(model_name=config["embedding_model"])
	query_embedding = embedding_model.get_query_embedding(query)
	#timer 1 end
	vector_store_query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=config["top_k"], mode=config["query_mode"])
	api_key = "a812b3f1-6a93-4899-939e-13d5ee0e132e"
	pinecone_index = pinecone.Index("quickstart")
	vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
	#timer 2 start
	query_result = vector_store.query(vector_store_query)
	#timer 2 end
	#timer 3 start
	nodes_with_scores = []
	for index, node in enumerate(query_result.nodes):
		score: Optional[float] = None
		if query_result.similarities is not None:
			score = query_result.similarities[index]
		nodes_with_scores.append(NodeWithScore(node=node, score=score))
	#timer 3 end
	#timer 4 start
	USER_PROMPT = format_user_prompt(query,nodes_with_scores)
	response = gpt_helper(USER_PROMPT,system_prompt)
	#timer 4 end
	return [response,conversation_id]
