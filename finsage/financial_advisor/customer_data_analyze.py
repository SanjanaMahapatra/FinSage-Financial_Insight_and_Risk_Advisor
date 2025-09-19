from openai import  OpenAI

from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.llms import OpenAI
from langchain_openai import OpenAIEmbeddings

import pandas as pd
import io
import os
from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# initialise openai client
client = OpenAI(
  api_key= OPENAI_API_KEY,
)


# This function is analysing the transaction category of the users based on the credit and debit type

def categorize_transaction(description):
    response = client.chat.completions.create(
      model='gpt-4o-mini', 
      max_tokens=20, 
      temperature=0.5,
      messages=[
          {
            "role": "system",
            "content": """You are an expert auditor, skilled at categorizing transactions in the categories: 
            Income (Salary, bonuses, caskbacks etc.), Investments, Utilities, Bills, Transport, Fuel/Gas/Auto, Food (groceries, dining etc.), Shopping, Medical, Insurance, Entertainment, Travel, Education, Card Bills, Misc.
            The transactions can have credits (marked as 'CR') or debits (marked as 'DR'). 'CR' can not be categorized into outgoing balance categories, and DR can not be categorized into incoming categories.
            Note: Strictly only reply with one word - the respective category of transaction."""
          },
          {
            "role": "user",
            "content": description
          }
        ]                                   
      )
    return response.choices[0].message.content


def preprocess_df(df):
    df = df.rename(columns={'0': 'date', '1': 'description', '2': 'amount', '3': 'type'})
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['amount'] = df['amount'].replace(',', '').astype(float)
    df['description'] = df['description'].str.lower()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    return df


# dataframes = []
# for blob in blobs_list:
#     blob_client = container_client.get_blob_client(blob.name)
#     download_stream = blob_client.download_blob().readall()
#     df = pd.read_csv(io.BytesIO(download_stream))
#     dataframes.append(df)

# dataframes = [preprocess_df(df) for df in dataframes]
# }}} 
# {{{ def: get_monthly_insights

def get_monthly_insights(df):
    monthly_spend = df.groupby(['year', 'month', 'category'])['amount'].sum().reset_index()
    return monthly_spend


def write_csv(df, filename):
    df.to_csv(f'artifacts/{filename}.csv')
    # for df in dataframes:
    #   df.to_csv('artifacts/cat.csv', index=False)
# }}} 
# for df in dataframes:
#     df['category'] = df.apply(lambda row: categorize_transaction(' '.join(row.astype(str))), axis=1)

# load data
cust_data = pd.read_csv(r'./artifacts/cat.csv')
user_info = pd.read_csv(r'./artifacts/user_info.csv')

agent = create_csv_agent(
    OpenAI(api_key=OPENAI_API_KEY, temperature=0.5, model="gpt-4o-mini"),
    ['artifacts/user_info.csv', 'artifacts/cat.csv', 'artifacts/monthly_analysis.csv'],
    verbose=True,
    allow_dangerous_code=True
)

