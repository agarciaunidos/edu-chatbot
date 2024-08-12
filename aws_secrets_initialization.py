
import boto3
import json
from langchain.memory import DynamoDBChatMessageHistory
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
import streamlit as st

# Access the secrets
aws_access_key_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]



# Constants for configuration
REGION_NAME = 'us-east-1'
REGION_NAME_BEDROCK = 'us-east-1'
INDEX_NAME = 'edu-application-guide-full'
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
MODEL_ID_OPUS = 'anthropic.claude-3-opus-20240229-v1:0'
SESSION_TABLE_NAME = "SessionTable"
SESSION_ID = "99"
FUNCTION_NAME = 'policy-app-prompt-pinecone'

# Use the credentials to create an AWS session
aws_session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name =REGION_NAME
)

# Setup AWS boto3 session and clients
#aws_session = boto3.session.Session(region_name=REGION_NAME)
secrets_manager_client = aws_session.client(service_name='secretsmanager')
bedrock_client = boto3.client("bedrock-runtime", region_name=REGION_NAME_BEDROCK)
lambda_client = boto3.client('lambda', region_name=REGION_NAME, boto3_session = aws_session) 

# Initialize Langchain components
dynamodb_history = DynamoDBChatMessageHistory(table_name=SESSION_TABLE_NAME, session_id=SESSION_ID, boto3_session=aws_session)
embeddings = BedrockEmbeddings(client=bedrock_client, region_name=REGION_NAME_BEDROCK,model_id="amazon.titan-embed-text-v2:0" )
llm = ChatBedrock(model_id=MODEL_ID, region_name=REGION_NAME_BEDROCK, client=bedrock_client)

def fetch_secret_value(secret_name, key):
    """
    Fetches a secret value from AWS Secrets Manager by secret name and key.
    Returns the value associated with the key or None if the key doesn't exist.
    """
    try:
        response = secrets_manager_client.get_secret_value(SecretId=secret_name)
        if 'SecretString' in response:
            secret_json = json.loads(response['SecretString'])
            return secret_json.get(key)  # Safely return the value or None
    except Exception as e:
        print(f"Error retrieving the key '{key}' from the secret '{secret_name}': {e}")
        return None

# Retrieve API keys from Secrets Manager

PINECONE_API_KEY = fetch_secret_value("edu-app-secrets","PINECONE_API_KEY")
COHERE_API_KEY = fetch_secret_value("policy-app-secrets","COHERE_API_KEY")
#ALB_ARN = fetch_secret_value("policy-app-secrets", "ALB_ARN")
