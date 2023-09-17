from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Pinecone
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.output_parsers import ResponseSchema, StructuredOutputParser, RetryWithErrorOutputParser
import langchain
import pinecone
import os

def build_query_chain(llm):

    query_string = ResponseSchema(
        name="query_string",
        description="The string used to query the vector database for wine options.",
    )

    output_parser = StructuredOutputParser.from_response_schemas(
        [query_string]
    )
    response_format = output_parser.get_format_instructions()

    ## Issue and Summary Chain
    prompt = ChatPromptTemplate.from_template(
        """You are an expert wine sommelier. Your goal is to select a wine from the database to recommend to the user.
        Take a breath and understand the following user preferences:
        taste: {taste}
        experience level:{experience}
        wine color: {wine_color}
        flavor: {flavor}
        pairing: {pairing}
        complement: {complement}

        Now create a string that will be used to do a similarity search on a vector database containing wine descriptions.
        To build better queries for similarity search, ensure they are specific, utilize relevant features or descriptors.
        {response_format}
        """
    )

    small_chain = LLMChain(llm=llm, prompt=prompt, output_key="query")

    chain = SequentialChain(
        chains=[small_chain],
        input_variables=["taste", "experience","wine_color","flavor","pairing","complement"] + ["response_format"],
        output_variables=["query"],
        verbose=False,
    )
    return chain, response_format, output_parser


def build_recommendation_chain(llm):

    recommendation = ResponseSchema(
        name="recommendation",
        description="The recommended wine Name.",
    )

    explanation = ResponseSchema(
        name="explanation",
        description="The explanation of the selected recommendation.",
    )

    output_parser = StructuredOutputParser.from_response_schemas(
        [explanation,recommendation]
    )
    response_format = output_parser.get_format_instructions()

    ## Issue and Summary Chain
    prompt = ChatPromptTemplate.from_template(
        """You are an expert wine sommelier. Your goal is to select a wine from the options bellow to recommend to the user.
        Take a breath and understand the following user preferences:
        taste: {taste}
        experience level:{experience}
        wine color: {wine_color}
        flavor: {flavor}
        pairing: {pairing}
        complement: {complement}

        Now take a breath and understand the wine options:
        Option 1:
        {wine_1}
        Option 2:
        {wine_2}
        Option 3:
        {wine_3}

        Now select the best wine to recommend to this user.

        {response_format}
        """
    )

    small_chain = LLMChain(llm=llm, prompt=prompt, output_key="recommendation")

    chain = SequentialChain(
        chains=[small_chain],
        input_variables = [
        "taste", "experience", "wine_color", "flavor", "pairing", "complement",
        "response_format", "wine_1", "wine_2", "wine_3"],
        output_variables=["recommendation"],
        verbose=False,
    )
    return chain, response_format, output_parser
