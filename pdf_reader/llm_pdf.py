from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# async def a_invoke_model(msgs, schema, model="gpt-4o-mini"):
#         """Invoke the LLM model"""
#         gpt = ChatOpenAI(model=model, temperature=0.1).with_structured_output(schema=schema, strict=True)


#         result = await gpt.ainvoke(msgs)
    
#         # Estrai i token usage
#         if isinstance(result, dict) and 'raw' in result:
#             usage = result['raw'].response_metadata.get('token_usage', {})
#             print(f"\n📊 Token Usage:")
#             print(f"  Input tokens:  {usage.get('prompt_tokens', 0)}")
#             print(f"  Output tokens: {usage.get('completion_tokens', 0)}")
#             print(f"  Total tokens:  {usage.get('total_tokens', 0)}")
            
#             # Restituisci solo i dati parsati
#             return result['parsed']
        
#         return result
#         #return await gpt.ainvoke(msgs)


 
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
 
def get_token():
    import requests
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
            temperature=0,
    ):
        self.model_name = model_name
        token = get_token()
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Ocp-Apim-Subscription-Key": os.getenv("OCP_APIM_SUBSCRIPTION_KEY")
        }
        self.openai_api_base = os.getenv("OPENAI_BASE_URL_LLM")
        
        self.temperature = temperature
        try:
            self._client = ChatOpenAI(
                model=self.model_name,
                default_headers=self.headers,
                openai_api_base=self.openai_api_base,
                temperature=self.temperature,
                api_key="useless",
            )
        except Exception as e:
            print(f"[LLMClient] initialization error: {e}")
            self._client = None
   
    async def a_invoke_model(self, msgs):
        """Invoke the LLM model with structured output"""
        if self._client is None:
            raise ValueError("LLM Client non inizializzato correttamente")
       
        # Configura structured output

        # Invoca il modello
        result = await self._client.ainvoke(msgs)
       
        # Estrai i token usage se disponibili
        if hasattr(result, '__dict__'):
            usage_info = getattr(result, 'usage_metadata', None) or \
                        getattr(result, 'response_metadata', {}).get('token_usage', {})
           
            if usage_info:
                # print(f"\n📊 Token Usage:")
                # print(f"  Input tokens:  {usage_info.get('input_tokens', usage_info.get('prompt_tokens', 0))}")
                # print(f"  Output tokens: {usage_info.get('output_tokens', usage_info.get('completion_tokens', 0))}")
                print(f"  Total tokens:  {usage_info.get('total_tokens', 0)}")
       
        return result.content