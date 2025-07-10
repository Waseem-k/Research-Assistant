from qdrant_client import QdrantClient
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from jproperties import Properties
from ollama import Client as OllamaClient
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import ArxivRetriever
from langchain_ollama.llms import OllamaLLM
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain import hub
import numpy as np
from trulens.core import Feedback
from trulens.core import TruSession
from trulens.apps.langchain import TruChain
from trulens.core.feedback.feedback import Feedback
from trulens.providers.litellm import LiteLLM
from trulens.core.schema import Select
from trulens.dashboard import run_dashboard
import streamlit as st
session = TruSession()
session.reset_database()
# Run the TruLens dashboard
run_dashboard(
    session=session,
    port=8502,
)
import nest_asyncio
nest_asyncio.apply()

import warnings
warnings.filterwarnings('ignore')

#Read configurations from properties file
configs = Properties()
with open('app_config.properties', 'rb') as config_file:
    configs.load(config_file)
@st.cache_resource
def initialize_app_resources():
# Initialize Qdrant client
    qdrant_client = QdrantClient(
        url=configs.get("QDRANT_URL").data,
        api_key=configs.get("QDRANT_API_KEY").data
    )


    # Initialize Ollama client
    ollama_client = OllamaClient(host="http://localhost:11434")

    #Initialize ollama LLM
    llm = OllamaLLM(
        model="llama3",
        client=ollama_client,
        temperature=0.5,
        max_tokens=5000,

    )
    # Initialize Ollama embeddings
    embed = OllamaEmbeddings(
        model="llama3"
    )

    # Initialize vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=configs.get("QDRANT_COLLECTION_NAME").data,
        embedding=embed,
        content_payload_key="text"
    )

    # Qdrant retriever (1st retriever)
    qdrant_retriever = vector_store.as_retriever()

    # Arxiv retriever (2nd retriever)
    arxiv_retriever = ArxivRetriever(
        load_max_docs=3,
        get_full_documents=True,
    )

    # Ensemble retriever combining Qdrant and Arxiv retrievers
    combined_retriever = EnsembleRetriever(
        retrievers=[qdrant_retriever, arxiv_retriever],
        weights=[0.5, 0.5]  # Adjust weights as needed
    )

    #Initialize llm provider for TruLens
    ollama_provider = LiteLLM(
        model_engine="ollama/llama3", api_base="http://localhost:11434"
    )

    #Set context for TruLens
    qdrant_context = (
        Select.RecordCalls.retrievers[0]
        ._get_relevant_documents.rets[:]
        .page_content
    )
    arvix_context = (
        Select.RecordCalls.retrievers[1]
        ._get_relevant_documents.rets[:]
        .page_content
    )
    ensemble_context = Select.RecordCalls.invoke.rets[:].page_content

    # Setup TruLens Feedback
    #Context Relevance Feedback for Qdrant, Arxiv, and Ensemble retrievers
    f_context_relevance_qdrant = (
        Feedback(ollama_provider.context_relevance_with_cot_reasons, name="qdrant")
        .on_input()
        .on(qdrant_context)
        .aggregate(np.mean)
    )

    f_context_relevance_arvix = (
        Feedback(ollama_provider.context_relevance_with_cot_reasons, name="arvix")
        .on_input()
        .on(arvix_context)
        .aggregate(np.mean)
    )

    f_context_relevance_ensemble = (
        Feedback(ollama_provider.context_relevance_with_cot_reasons, name="Ensemble")
        .on_input()
        .on(ensemble_context)
        .aggregate(np.mean)
    )

    #Groundness Feedback function
    f_groundedness_qdrant = (
        Feedback(ollama_provider.groundedness_measure_with_cot_reasons, name="Groundedness Qdrant"
        )
        .on(qdrant_context.collect())  # collect context chunks into a list
        .on_output()

    )
    f_groundedness_arvix = (
        Feedback(ollama_provider.groundedness_measure_with_cot_reasons, name="Groundedness Arvix"
        )
        .on(arvix_context.collect())  # collect context chunks into a list
        .on_output()
    )
    f_groundedness = (
        Feedback(ollama_provider.groundedness_measure_with_cot_reasons, name="Groundedness"
        )
        .on(ensemble_context.collect())  # collect context chunks into a list
        .on_output()
    )

    # Answer Relevance Feedback
    f_answer_relevance = Feedback(
        ollama_provider.relevance_with_cot_reasons, name="Answer Relevance"
    ).on_input_output()

    # Initialize the rag_chain
    prompt = hub.pull("rajkstats/science-product-rag-prompt-reasoning")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    # def format_titles_with_links(docs):
    #     links = []
    #     for doc in docs:
    #         title = doc.metadata.get("Title", "Untitled")
    #         url = doc.metadata.get("link", "#")
    #         links.append(f"[{title}]({url})")
    #     return "\n".join(links)

    # Create a RAG chain with the combined retriever and LLM
    rag_chain = (
        {"retrieved_documents": combined_retriever | format_docs, "user_query": RunnablePassthrough()}
        | prompt
        | llm
        # |(lambda x: {"answer": x, "context": format_titles_with_links(x.get("retrieved_documents", []))})
        | StrOutputParser()
    )

    tru_recorder = TruChain(
        combined_retriever,
        app_name="Research Assistant",
        app_version="v1.0",
        selectors_nocheck=True,
        feedbacks=[
            f_context_relevance_qdrant,
            f_context_relevance_arvix,
            f_context_relevance_ensemble,
            f_groundedness_qdrant,
            f_groundedness_arvix,
            f_groundedness,
            f_answer_relevance,
        ],
    )
    return rag_chain, tru_recorder
# Function to run user query
if __name__ == "__main__":
    # Wrap the logic in streamlit ui
    st.set_page_config(page_title="🧠 Research Assistant", layout="wide")
    st.title("📚 RAG-based Research Assistant")
    st.subheader("Upload research papers and ask questions related to them.")
    rag_chain, tru_recorder = initialize_app_resources()
    user_query = st.text_input("Enter your query (or 'exit' to quit): ")
    if user_query:
        if user_query.lower() == 'exit':
            st.stop()
        with tru_recorder as recording:
            response = rag_chain.invoke(user_query)
            st.write("Response:", response)
            #st.write("Source Documents:", response.get("format_docs", "No context found."))