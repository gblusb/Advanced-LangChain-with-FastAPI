import os

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from models import DocumentModel, DocumentResponse
from store import AsnyPgVector
from store_factory import get_vector_store

load_dotenv(find_dotenv())

app = FastAPI()


def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not found.")
    return value


try:
    USE_ASYNC = os.getenv("USE_ASYNC", "False").lower() == "true"
    if USE_ASYNC:
        print("Async project used")

    POSTGRES_DB = get_env_variable("POSTGRES_DB")
    POSTGRES_USER = get_env_variable("POSTGRES_USER")
    POSTGRES_PASSWORD = get_env_variable("POSTGRES_PASSWORD")
    DB_HOST = get_env_variable("DB_HOST")
    DB_PORT = get_env_variable("DB_PORT")

    CONNECTION_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"

    OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()

    # 벡터 저장소를 생성: 문서를 저장하고 검색하는데 사용
    mode = "async" if USE_ASYNC else "sync"
    pgvector_store = get_vector_store(
        connection_string=CONNECTION_STRING,
        embeddings=embeddings,
        collection_name="testcollection", # 벡터저장소 생성시 사용할 컬렉션 이름(컬렉션은 데이터를 그룹화함)
        mode=mode,
    )
    # 질문과 관련이 높은 문서를 벡터저장소에서 검색하는 역할을 하는 retriever 객체 생성
    retriever = pgvector_store.as_retriever()
    # 템플릿 설정, 질문(question)과 관련된 문맥(context)을 입력으로 받아 응답을 생성함
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    # 템플릿을 사용하여 질문을 처리함
    prompt = ChatPromptTemplate.from_template(template)
    # 질문에 대한 응답 생성
    model = ChatOpenAI(model_name="gpt-3.5-turbo")
    # 체인: 질문을 입력받아 응답을 생성하고 응답을 처리하여 최종적인 응답으로 변환함
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


except ValueError as e:
    raise HTTPException(status_code=500, detail=str(e))
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-documents/")
async def add_documents(documents: list[DocumentModel]):
    try:
        docs = [
            Document(
                page_content=doc.page_content,
                metadata=(
                    {**doc.metadata, "digest": doc.generate_digest()}
                    if doc.metadata
                    else {"digest": doc.generate_digest()}
                ),
            )
            for doc in documents
        ]
        ids = (
            await pgvector_store.aadd_documents(docs)
            if isinstance(pgvector_store, AsnyPgVector)
            else pgvector_store.add_documents(docs)
        )
        return {"message": "Documents added successfully", "ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-all-ids/")
async def get_all_ids():
    try:
        if isinstance(pgvector_store, AsnyPgVector):
            ids = await pgvector_store.get_all_ids()
        else:
            ids = pgvector_store.get_all_ids()

        return ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-documents-by-ids/", response_model=list[DocumentResponse])
async def get_documents_by_ids(ids: list[str]):
    try:
        if isinstance(pgvector_store, AsnyPgVector):
            existing_ids = await pgvector_store.get_all_ids()
            documents = await pgvector_store.get_documents_by_ids(ids)
        else:
            existing_ids = pgvector_store.get_all_ids()
            documents = pgvector_store.get_documents_by_ids(ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        return documents
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-documents/")
async def delete_documents(ids: list[str]):
    try:
        if isinstance(pgvector_store, AsnyPgVector):
            existing_ids = await pgvector_store.get_all_ids()
            await pgvector_store.delete(ids=ids)
        else:
            existing_ids = pgvector_store.get_all_ids()
            pgvector_store.delete(ids=ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        return {"message": f"{len(ids)} documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/")
async def quick_response(msg: str):
    result = chain.invoke(msg)
    return result
