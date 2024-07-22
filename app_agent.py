"""
A simple LSEG HR agent based on AI Dial, RAG and langchain
"""

import os
from typing import List
from urllib.parse import urljoin
from uuid import uuid4

import uvicorn
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import CacheBackedEmbeddings
from langchain.globals import set_debug
from langchain.chains import create_retrieval_chain
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import get_last_attachment_url, sanitize_namespace

from aidial_sdk import DIALApp
from aidial_sdk import HTTPException as DIALException
from aidial_sdk.chat_completion import ChatCompletion, Choice, Request, Response


def get_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Please provide {name!r} environment variable")
    return value


# get_env("DIAL_URL")
DIAL_URL = "http://localhost:3000"
AZURE_AI_URL = os.getenv("AZURE_AI_URL", "https://ai-proxy.lab.epam.com")
API_KEY = get_env("API_KEY")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-ada-002")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")
API_VERSION = os.getenv("API_VERSION", "2024-02-01")
LANGCHAIN_DEBUG = os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"

set_debug(LANGCHAIN_DEBUG)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=256, chunk_overlap=0
)

embedding_store = LocalFileStore("./cache/")


class CustomCallbackHandler(AsyncCallbackHandler):
    def __init__(self, choice: Choice):
        self._choice = choice

    async def on_llm_new_token(self, token: str, *args, **kwargs) -> None:
        self._choice.append_content(token)


# load the document and split it into chunks
def process(file_abs_url: str, choice: Choice) -> list[Document]:
    if file_abs_url.endswith(".pdf"):
        loader = PyPDFLoader(file_abs_url)
    else:
        loader = WebBaseLoader(file_abs_url)

    # Create the download stage to show to the user the active process.
    # After the loading is complete, the stage will auto finished.
    with choice.create_stage("Downloading the document"):
        try:
            documents = loader.load()
        except Exception:
            msg = "Error while loading the document. Please check that the URL you provided is correct."
            raise DIALException(
                status_code=400, message=msg, display_message=msg
            )

    # Show the user the total number of parts in the resource
    with choice.create_stage(
            "Splitting the document into chunks"
    ) as stage:
        texts = text_splitter.split_documents(documents)
        stage.append_content(f"Total number of chunks: {len(texts)}")
    return texts


class LsegHRRAGApplication(ChatCompletion):

    async def chat_completion(
            self, request: Request, response: Response
    ) -> None:
        collection_name = str(uuid4())

        with response.create_single_choice() as choice:
            message = request.messages[-1]
            user_query = message.content or ""

            # declare file url array of file urls
            file_urls = ["api/files/FSWLtFA648cQNf6WfxHZcFzdABKNsTr7ygwQjYbiDi1n/Onboarding%20Guide.pdf",
                         "api/files/FSWLtFA648cQNf6WfxHZcFzdABKNsTr7ygwQjYbiDi1n/LSEG%20Background%20Check.pdf",
                         "api/files/FSWLtFA648cQNf6WfxHZcFzdABKNsTr7ygwQjYbiDi1n/Onboarding%20Within%20LSEG%20account.pdf",
                         "api/files/FSWLtFA648cQNf6WfxHZcFzdABKNsTr7ygwQjYbiDi1n/New%20Joiner%20General%20Workflow.txt"]

            # declare empty list[Document] texts
            texts = []
            # for each file url in file urls get file abs url as ulrjoin DIAL_URL and file url
            for file_url in file_urls:
                file_abs_url = urljoin(f"{DIAL_URL}/", file_url)
                # process the file abs url and append the result to texts
                texts.extend(process(file_abs_url, choice))

            # Show the user start of calculating embeddings stage
            with choice.create_stage("Calculating embeddings"):
                openai_embedding = AzureOpenAIEmbeddings(
                    model=EMBEDDINGS_MODEL,
                    azure_deployment=EMBEDDINGS_MODEL,
                    azure_endpoint=AZURE_AI_URL,
                    openai_api_key=API_KEY,
                    openai_api_version=API_VERSION,
                    # The check leads to tokenization of the input strings.
                    # Tokenized input is only supported by OpenAI embedding models.
                    # For other models, the check should be disabled.
                    check_embedding_ctx_length=False,
                )

                embeddings = CacheBackedEmbeddings.from_bytes_store(
                    openai_embedding,
                    embedding_store,
                    namespace=sanitize_namespace(openai_embedding.model),
                )

                docsearch = Chroma.from_documents(
                    texts, embeddings, collection_name=collection_name
                )

            # CustomCallbackHandler allows to pass tokens to the users as they are generated, so as not to wait for a
            # complete response.
            llm = AzureChatOpenAI(
                azure_deployment=CHAT_MODEL,
                azure_endpoint=AZURE_AI_URL,
                openai_api_key=API_KEY,
                openai_api_version=API_VERSION,
                temperature=0,
                streaming=True,
                callbacks=[CustomCallbackHandler(choice)],
            )

            await response.aflush()
            system_prompt = (
                "Context: {context}"
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", user_query),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            
            chain = create_retrieval_chain(docsearch.as_retriever(search_kwargs={"k": 15}), question_answer_chain)
            # qa = RetrievalQA.from_chain_type(
            #    llm=llm,
            #    chain_type="stuff",
            #    retriever=docsearch.as_retriever(search_kwargs={"k": 15}),
            # )

            await chain.ainvoke({"input": user_query})

            docsearch.delete_collection()


app = DIALApp(DIAL_URL, propagation_auth_headers=True)
app.add_chat_completion("lseg-hr", LsegHRRAGApplication())

if __name__ == "__main__":
    uvicorn.run(app, port=5002)
