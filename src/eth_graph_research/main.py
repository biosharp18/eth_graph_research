"""Auth check: first run opens a Google login; later runs use cached credentials."""

import pydata_google_auth
from google.cloud import bigquery

PROJECT_ID = "eth-graph-research"

credentials = pydata_google_auth.get_user_credentials(
    scopes=["https://www.googleapis.com/auth/bigquery"],
)
client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

query = """
SELECT table_name
FROM `bigquery-public-data.goog_blockchain_ethereum_mainnet_us.INFORMATION_SCHEMA.TABLES`
ORDER BY table_name
"""
df = client.query(query).to_dataframe()
print(df)
