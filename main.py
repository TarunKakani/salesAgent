import pandas as pd
from typing import List, Tuple
import math
from langchain.tools import tool
from langchain_community.llms import LlamaCpp
from langchain.agents import initialize_agent, AgentType

## File Loading
current_df = pd.read_csv("SalesDataset.csv")
historical_df = pd.read_csv("SalesDatasetSheet2.csv")

# FILE = "sales_dataset.xlsx"
# current_df = pd.read_excel(FILE, sheet_name="CurrentYear")
# historical_df = pd.read_excel(FILE, sheet_name="HistoricalYear")

# current_df["Date"] = pd.to_datetime(current_df["Date"])
# historical_df["Date"] = pd.to_datetime(historical_df["Date"])

# current_df.to_csv("SalesDataset", index=True)
# historical_df.to_csv("SalesDatasetSheet2.csv", index=True)


## Functions(Queries) for agent - error handling is not there
# Low Complexity
def total_sales_product(product: str) -> float: # this syntax is called type hints, better code better readiblity
    return current_df.loc[current_df["Product"] == product, "Total_Sales"].sum()

def count_transactions_exact_qty(qty: int) -> int:
    return int((current_df["Quantity"] == qty).sum())

def average_price_product(product: str) -> float:
    return current_df.loc[current_df["Product"] == product, "Price"].mean()

def max_single_transaction() -> float:
    return current_df["Total_Sales"].max()

def count_transactions_month(year:int, month:int) -> int:
    mask = (current_df["Date"].dt.year == year) & (current_df["Date"].dt.month == month)
    return int(mask.sum())

## Medium Complexity
# Compares total sales for a product in Q1 vs Q2
def compare_total_sales_q1_q2(product) -> Tuple[float, float]:
    q1 = current_df[(current_df["Date"].dt.quarter == 1) & (current_df["Product"] == product)]["Total_Sales"].sum()
    q2 = current_df[(current_df["Date"].dt.quarter == 2) & (current_df["Product"] == product)]["Total_Sales"].sum()
    return q1,q2

# Identifies the product with largest percentage sales increase compared to historical data
def biggest_percent_increase() -> Tuple[str, float]:
    curr = current_df.groupby("Product")["Total_Sales"].sum()
    hist = historical_df.groupby("Product")["Total_Sales"].sum()
    pct = ((curr - hist) / hist.replace(0, math.nan)).dropna() # replace zero values with nan(null) and drop those values (delete rows)
    top = pct.idxmax()
    return str(top), pct.loc[top] * 100

## Tools
@tool
def ask_low(query: str) -> str:
    """Handles low-complexity queries about the sales dataset."""
    query = query.lower()
    if "highest single transaction" in query or "highest query" in query:
        return f"Highest single transaction = ${max_single_transaction():.0f}"
    
    # not working
    elif "average price for product" in query:
        product = query.split("product")[-1].strip()
        avg_price = average_price_product(product)
        return f"Average price for {product} = ${avg_price:.2f}" if not math.isnan(avg_price) else f"No data for {product}"
    
    return "I can't answer that low-level question yet!"

## ask_med, ask_high

## Plug an open source LLM via llama-cpp and initiate the agent
MODEL_PATH = "/Users/watchdog/Documents/mL/agent/models/Phi-3-mini-4k-instruct-q4.gguf"  # Model Path

# Model Setup
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=4096,
    temperature=0.1,
    verbose=False,
    n_gpu_layers=-1,
)

# Initialize Agent
TOOLS = [ask_low]
agent = initialize_agent(
    tools=TOOLS,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,  # Limit retries to prevent infinite loop
)

# Execution (Queries)
if __name__ == "__main__":
    queries = [
        "What is the highest single transaction?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = agent.run(query)
        print(f"Response: {response}")