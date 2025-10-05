from fin_data_qa import analyze_financial_data
from vector_db_qa import query_vector_db

from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import yfinance as yf

from pydantic import BaseModel, Field
from typing import List
import pdfplumber
from PyPDF2 import PdfReader
import openai

import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# class for financial data analysis

class FinancialDataAnalysisInput(BaseModel):
    """input for Financial data analysis"""

    query: str = Field(..., description = 'query string for financial data analysis')

class FinancialDataAnalysisTool(BaseTool):
    name : str = 'analyze_financial_data'
    description : str = """useful when you need to look up financial data for the user, their spending habits, retrieve their balance, look up their financial goals and risk profile, visualize data etc. Useful for any task that requires user's personal data.
    Note: Always provide concise, accurate and invormative, well formatted response, in bullets whenever possible."""

    def _run(self, query: str):
        with st.spinner('Looking up your financial data...'):
            fin_data_analysis_response = analyze_financial_data(query)
            st.success('✅ Analyzed your financial data')

        return fin_data_analysis_response

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = FinancialDataAnalysisInput

# creating class for querying the vector db

class RecommendProductsInput(BaseModel):
    """input for Recommend product by querying vector database"""

    query: str = Field(..., description = 'query striing to find relevant products from vector database')

class RecommendProductsTool(BaseTool):
    name : str = 'query_vector_db'
    description : str = """Useful when you want to recommend products or answer any Bank of Baroda specific questions.
    Note: always provide accurate and invormative, well formatted response."""

    def _run(self, query: str):
        with st.spinner('Looking up BoB database...'):
            query_vector_db_response = query_vector_db(query)
            st.success('✅ Fetched relevant info from BoB database')

        return query_vector_db_response

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = RecommendProductsInput


# defining the class for yahoo finance api

def get_stock_price(symbol):
    ticker = yf.Ticker(symbol)
    try:
        todays_data = ticker.history(period='1d')
        return round(todays_data['Close'][0], 2)
    except:
        return 0

# class for defining Stock price lookup tool 

class StockPriceLookupInput(BaseModel):
    """input for Stock price look-up tool"""

    symbol: str = Field(..., description='ticker symbol to look up stocks price')

class StockPriceLookupTool(BaseTool):
    name : str = 'get_stock_price'
    description : str = 'useful when you want to look up price of a specific company using its ticker/symbol'

    def _run(self, symbol: str):
        with st.spinner('Looking up stock price...'):
            stock_price_response = get_stock_price(symbol)
            st.success('✅ Looked up stock price')

        return stock_price_response

    def _arun(self, symbol: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceLookupInput

# def class for google search

tavily_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY, max_results=20)
def search_investment_options(search_query):
    search_response = tavily_tool.invoke(search_query)
    
    return search_response

# defining class for various search investment options/stocks

class SearchInvestmentOptionsInput(BaseModel):
    """input for Search Investment Options/Stocks"""
    search_query: str = Field(..., description = "search query to search google for investment options/stocks based on user's risk profile and investment goals")

class SearchInvestmentOptionsTool(BaseTool):
    name: str = 'search_investment_options'
    description : str = """useful when you want to search google to suggest investment options/list of good stocks, based on user's risk profile and investment goals. 
    Note:
    - Always reformat the input query in a way that will give the best search results.
    - Search strictly for India specific. 
    - In case of stocks, return stock's actual ticker/symbol. 
    - You can access user's risk profile and goals by calling 'analyze_financial_data'
    - Attempt to make the responses Bank of Baroda centric if possible.
    - Post-search, always call 'query_vector_db' to search for relevant products and also recommend them along the answer.
    """
    
    def _run(self, search_query: str):
        with st.spinner('Searching google for investment options...'):
            search_response = search_investment_options(search_query)
            st.success('✅ Searched google for investment options')
        return search_response

    def _arun(self, search_query: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = SearchInvestmentOptionsInput


tools = [FinancialDataAnalysisTool(), RecommendProductsTool(), StockPriceLookupTool(), SearchInvestmentOptionsTool()] # pyright: ignore[reportCallIssue]

llm = ChatOpenAI(temperature=0, model='gpt-4o-mini')

open_ai_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# streamlit config

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title='Finsage - Financial Insights and Risk Advisor', page_icon='🏆', layout='wide')


with st.sidebar:
    st.subheader("Helping you with financial insights!")
    st.divider()
    
    st.markdown(
        """
        Hello! As your financial advisor at Bank of Baroda, I can assist you with a variety of financial services and information, including:
        📊 **Features**:  
        - AI-powered **Financial Advice**  
        - **Financial Analysis**: Review your spending habits, savings, and investments.
        - **Investment Options**: Recommend stocks, mutual funds, and other investment opportunities based on your risk profile and financial goals.
        - **Product Recommendations**: Suggest relevant banking products such as loans, savings accounts, and fixed deposits.
        - **Market Insights**: Provide updates on stock prices and market trends.
        - **Financial Planning**: Help you set and achieve your financial goals.
        """
    )
    st.divider()
    theme = st.radio("🌗 Select Theme:", ["Light Mode 🌞", "Dark Mode 🌙"])
    st.divider()
    
# --- Apply Custom Styling Based on Theme ---
is_dark = theme == "Dark Mode 🌙"
text_color = "#FFFFFF" if is_dark else "#000000"
bg_color = "#1E1E1E" if is_dark else "#F5F5F5"
button_color = "#FF4B4B" if is_dark else "#008080"
button_text_color = "#FFFFFF"
checkbox_text_color = "#FFFFFF" if is_dark else "#000000"

custom_css = f"""
    <style>
        .stApp {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        .stTextInput, .stFileUploader, .stTextArea {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        .stButton>button {{
            background-color: {button_color} !important;
            color: {button_text_color} !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 8px 15px !important;
        }}
        .stMarkdown, .stTextArea, .stSuccess {{
            color: {text_color} !important;
        }}
        label, div[data-testid="stCheckbox"] > label {{
            color: {checkbox_text_color} !important;
            font-weight: bold;
        }}
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# extracting the text from the pdf uploaded

def extract_text_from_pdf(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# --- Redact Sensitive Data Function ---
def redact_sensitive_info(text):
    redacted_text = text.replace("SSN", "[REDACTED]").replace("Account Number", "[REDACTED]")  # Customize as needed
    return redacted_text


from langchain_core.prompts import ChatPromptTemplate

def query_finetuned_gpt(query, chat_history):
    
    # Creating the prompt template
    template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial advisor at Bank of Baroda, assisting users with informative and accurate answers.
        You are a financial advisory assistant helping users with financial analysis and tax planning.

        Note:
        - Always provide concise, accurate and informative, well formatted response, in bullets whenever possible/needed.
        - Make sure never share specific user information (phone number, email id etc.) apart from names."""),
        
        ("human", "Chat history: {chat_history}\nUser question: {user_question}")
    ])
    
    
    formatted_messages = template.invoke({
        "chat_history": chat_history, 
        "user_question": query
    }).to_messages()
    
    messages = []
    for msg in formatted_messages:
        messages.append({
            "role": "system" if msg.type == "system" else "user",
            "content": msg.content
        })
    
    # Make API call
    with st.spinner('Processing...'):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5
        )
    
    return response.choices[0].message.content


# get response
def get_response(query, chat_history):
    template = """You are an expert financial advisor at Bank of Baroda, assisting users with informative and accurate answers.

    Chat history: {chat_history}
    User question: {user_question}

    Note:
    - Always provide concise, accurate and invormative, well formatted response, in bullets whenever possible/needed.
    - Make sure never share specific user information (phone number, email id etc.) apart from names.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | open_ai_agent
    with st.spinner('Processing...'):
        result = chain.invoke({"chat_history": chat_history, "user_question": query})
    return result['output']

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message('Human'):
            st.markdown(message.content)
    else:
        with st.chat_message('AI'):
            st.markdown(message.content)

template_questions = [
    "Analyze my spending habits and share ways to optimize",
    "Based on my goals, recommend some BoB products",
    "As per my risk profile, can you suggest some investment options?",
]


st.title('FINSAGE - Financial & Investment Advisory')

st.subheader("📢 Ask AI About Your Finances")
user_query = st.text_input(
    "Ask me anything related to finance, investments, or tax planning:",
    placeholder="E.g., What are some good investment strategies?"
)

if st.button("🔍 Ask AI"):
    if user_query:
        with st.spinner("🤖 Thinking..."):
            response = query_finetuned_gpt(user_query, st.session_state.chat_history)
        st.success("✅ Response Received")
        st.markdown(f"**🤖 AI Advisor:** <span style='color: {text_color};'>{response}</span>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a question before clicking the button.")

st.divider()


st.subheader("📤 Upload the Documents to analyse the financial insights")

col1, col2 = st.columns(2)

with col1:
    bank_txn_data = st.file_uploader("📄 Upload Your Bank Transaction Statements (PDF)", type=["pdf"])
with col2:
    medical_spendings = st.file_uploader("📜 Upload Your Medical Spendings (PDF)", type=["pdf"])
    
# --- Toggle Options (Now Clearly Visible in Light Mode) ---
retain_data = st.checkbox("🔄 Retain Data for Future Queries", key="retain", help="Keep extracted data for future analysis.")
redact_data = st.checkbox("🛑 Redact Sensitive Information", key="redact", help="Mask sensitive data before analysis.")


extracted_text = ""

if bank_txn_data or medical_spendings:
    st.success("✅ File(s) Uploaded Successfully!")

    # Extract text from both documents
    with st.spinner("📄 Extracting text from documents..."):
        if bank_txn_data:
            bank_txt_text = extract_text_from_pdf(bank_txn_data)
            if redact_data:
                bank_txt_text = redact_sensitive_info(bank_txt_text)
            extracted_text += f"Payslip:\n{bank_txt_text[:1000]}\n\n"

        if medical_spendings:
            medical_spendings_txt = extract_text_from_pdf(medical_spendings)
            if redact_data:
                medical_spendings_txt = redact_sensitive_info(medical_spendings_txt)
            extracted_text += f"Tax Form:\n{medical_spendings_txt[:1000]}\n\n"

    # Display Extracted Text Preview
    st.subheader("📄 Extracted Document Text (Preview)")
    st.text_area("📜 Document Data", extracted_text[:2000], height=250)

    # --- Analyze Document Button ---
    if st.button("📊 Analyze Financial Document"):
        if extracted_text:
            with st.spinner("🤖 Analyzing..."):
                analysis = query_finetuned_gpt(user_query, st.session_state.chat_history)
                # analysis = get_res(f"Analyze this financial document and provide insights:\n\n{extracted_text[:2000]}")
            st.success("✅ Analysis Completed")
            st.markdown("### 📊 AI-Powered Financial Insights:")
            st.text_area("📋 AI Analysis", analysis, height=300)
        else:
            st.warning("⚠️ No text extracted from the document.")


    # --- Forget Data If Not Retaining ---
    if not retain_data:
        payslip_text = ""
        tax_form_text = ""
        st.warning("🔒 Data is not stored for future use. AI responses will not be retained.")

st.markdown("Try out:")

cols = st.columns(len(template_questions))

for i, question in enumerate(template_questions):
    if cols[i].button(question):
        user_query = question
        st.session_state.chat_history.append(HumanMessage(user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)
        with st.chat_message("AI"):
            ai_response = get_response(user_query, st.session_state.chat_history)
            st.markdown(ai_response)
        st.session_state.chat_history.append(AIMessage(ai_response))

user_query = st.chat_input("Your message...")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = get_response(user_query, st.session_state.chat_history)
        st.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(ai_response))
    
    
st.divider()

# --- Footer ---
st.markdown("---")
st.markdown(f"🔒 **Your data is secure & private. AI-generated insights are for informational purposes only.**", unsafe_allow_html=True)

