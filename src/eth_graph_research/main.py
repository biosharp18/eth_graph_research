import pydata_google_auth
from google.cloud import bigquery

PROJECT_ID = "eth-graph-research"  # your actual project ID

credentials = pydata_google_auth.get_user_credentials(
    scopes=["https://www.googleapis.com/auth/bigquery"],
)
client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

query = """
SELECT symbol, name, decimals
FROM `bigquery-public-data.crypto_ethereum.tokens`
WHERE LOWER(address) = '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48'
"""
df = client.query(query).to_dataframe()
print(df)