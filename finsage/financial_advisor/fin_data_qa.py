import logging
import sys
from IPython.display import Markdown, display
import os
from dotenv import load_dotenv

import pandas as pd

from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Workflow as WF
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)

from llama_index.core import PromptTemplate

from pyvis.network import Network

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

# loading the data

cust_data = pd.read_csv("../artifacts/cat.csv")
user_data = pd.read_csv("../artifacts/user_info.csv")
user_data_dict = user_data.to_dict(orient='records')

# prompts

instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)


response_synthesis_prompt_str = (
    "You are a highly experienced Indian financial advisor, skilled are analysing customer financial data and recommending informative insights for investment.\n"
    "This is the customer information:\n"
    "{user_data_dict}"
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=cust_data.head(5),
)
pandas_prompt_user_data = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=user_data.head(5)
)

pandas_output_parser = PandasInstructionParser(cust_data)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str).partial_format(user_data_dict=user_data_dict)

# defining a pandas query engine
query_engine = PandasQueryEngine(
    df=cust_data,
    llm=llm,
    pandas_prompt=pandas_prompt,
    instruction_str=instruction_str,
    instruction_parser=pandas_output_parser,
    response_synthesis_prompt=response_synthesis_prompt,
    verbose=True
)


def analyze_financial_data(query):
    response = query_engine.query(query)
    print("Response coming from analyzing financial data ==> ", response)
    return response

