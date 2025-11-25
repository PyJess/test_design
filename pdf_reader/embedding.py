from typing import Optional, List, Union
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import requests

load_dotenv()

CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')


def get_token():

    url = os.getenv('BASE_URL_TOKEN')
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': 'api://3bb3eccb-0787-4526-811e-ec3dab677121/.default',
        'grant_type': 'client_credentials'
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    response = requests.post(url, data=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        print(f"Error Token: {response.status_code} - {response.text}")
        return None


class LLMClient:
    def __init__(
            self,
            model_name: str = "gpt-4o-mini",
            embedder_model_name: str = "text-embedding-3-large",
            temperature=0,
            # max_tokens=1000,
            # top_p=1
    ):
        self.model_name = model_name
        token = self._authenticate()
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Ocp-Apim-Subscription-Key": os.getenv("OCP_APIM_SUBSCRIPTION_KEY")
        }
        self.openai_api_base = os.getenv("OPENAI_BASE_URL")
        self.temperature = temperature
        self.embedding_deployment = embedder_model_name
        self.PROJECT_ID = os.getenv("PROJECT_ID", "default")
        try:
            self._client = ChatOpenAI(
                model=self.model_name,
                default_headers=self.headers,
                openai_api_base=self.openai_api_base + "/aw",
                temperature=self.temperature,
                api_key="useless",
            )

            self.embedder = OpenAIEmbeddings(
                model=self.embedding_deployment,  # just for metadata
                base_url=f"{self.openai_api_base}/deployments/{self.embedding_deployment}",
                default_headers=self.headers,
                api_key="useless",
            )
        except Exception as e:
            print(f"[LLMClient] initialization error: {e}")
            self._client = None

    def __getattr__(self, item):
        return getattr(self._client, item)

    def _authenticate(self) -> str:
        """Retrieve token."""
        return get_token()

    def available(self) -> bool:
        return self._client is not None

    def get_embedding(self, text:str)->Union[List[float], None]:

        """
        Returns the embedding vector for the given text.
        """

        url = f"{self.openai_api_base}/deployments/{self.embedding_deployment}/embeddings?project={self.PROJECT_ID}"

        payload = {
            "user_input": [text]
        }

        try:
            # Send request
            response = requests.post(url, headers=self.headers, json=payload)
            # Since we pass single query, we get index '0'
            return response.json()["output"][0]

        except Exception as e:
            print(f"[LLMClient] error getting embedding: {e}")
            return None