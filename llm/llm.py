# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# import os

# load_dotenv()

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





from langchain_openai import ChatOpenAI
from PIL import Image
from io import BytesIO
import base64
import os
from dotenv import load_dotenv


load_dotenv()

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
        self.openai_api_base = os.getenv("OPENAI_BASE_URL")
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
    
    async def a_invoke_model(self, msgs, schema):
        """Invoke the LLM model with structured output"""
        if self._client is None:
            raise ValueError("LLM Client non inizializzato correttamente")
        
        # Configura structured output
        llm_with_structure = self._client.with_structured_output(
            schema=schema, 
            strict=True
        )
        
        # Invoca il modello
        result = await llm_with_structure.ainvoke(msgs)
        
        # Estrai i token usage se disponibili
        if hasattr(result, '__dict__'):
            usage_info = getattr(result, 'usage_metadata', None) or \
                        getattr(result, 'response_metadata', {}).get('token_usage', {})
            
            if usage_info:
                print(f"\n📊 Token Usage:")
                print(f"  Input tokens:  {usage_info.get('input_tokens', usage_info.get('prompt_tokens', 0))}")
                print(f"  Output tokens: {usage_info.get('output_tokens', usage_info.get('completion_tokens', 0))}")
                print(f"  Total tokens:  {usage_info.get('total_tokens', 0)}")
        
        return result
    

    def process_images_from_folder(self, system_prompt: str, folder_path: str):
        """Process all PNG and JPEG images from a folder"""
        if self._openai_client is None:
            raise ValueError("OpenAI Client non inizializzato correttamente")
        
        try:
            
            print(f"Scanning folder: {folder_path}")
            
            # Check if folder exists
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            if not os.path.isdir(folder_path):
                raise ValueError(f"Path is not a folder: {folder_path}")
            
            # Supported image extensions
            supported_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
            
            # Get all image files from folder
            image_files = []
            for filename in sorted(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, filename)
                
                # Check if it's a file (not a subfolder)
                if os.path.isfile(file_path):
                    # Check extension
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in supported_extensions:
                        image_files.append(file_path)
            
            if not image_files:
                print(f"No image files found in folder: {folder_path}")
                return "No images found in the specified folder."
            
            print(f"Found {len(image_files)} image(s):")
            for img in image_files:
                print(f"  - {os.path.basename(img)}")
            
            # Prepare content
            content = [{"type": "text", "text": system_prompt}]
            
            for i, image_path in enumerate(image_files):
                print(f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
                
                try:
                    # Open image (PIL auto-detects format)
                    image = Image.open(image_path)
                    print(f"  Format: {image.format}, Size: {image.size}, Mode: {image.mode}")
                    
                    # Convert to RGB if necessary
                    if image.mode not in ('RGB', 'L'):
                        print(f"  Converting from {image.mode} to RGB")
                        image = image.convert('RGB')
                    
                    # Determine output format
                    output_format = "JPEG" if image.format in ["JPEG", "JPG"] else "PNG"
                    mime_type = "image/jpeg" if output_format == "JPEG" else "image/png"
                    
                    # Convert to base64
                    buffered = BytesIO()
                    if output_format == "JPEG":
                        image.save(buffered, format="JPEG", quality=95, optimize=True)
                    else:
                        image.save(buffered, format="PNG", optimize=True)
                    
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_base64}",
                            "detail": "high"
                        }
                    })
                    print(f"  ✓ Added as {output_format} (base64 length: {len(img_base64)})")
                    
                except Exception as e:
                    print(f"  ✗ Error processing {os.path.basename(image_path)}: {e}")
                    continue
            
            if len(content) == 1:
                print("\nWarning: No valid images were processed!")
                return "No images were successfully loaded."
            
            print(f"\n{'='*50}")
            print(f"Sending {len(content)-1} image(s) to OpenAI API...")
            print(f"{'='*50}")
            
            response = self._openai_client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[{"role": "user", "content": content}]
            )
            
            # Print token usage
            if response.usage:
                print(f"\n Token Usage:")
                print(f"  Input tokens:  {response.usage.prompt_tokens}")
                print(f"  Output tokens: {response.usage.completion_tokens}")
                print(f"  Total tokens:  {response.usage.total_tokens}")
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error processing folder: {e}")
            import traceback
            traceback.print_exc()
            raise


    def process_with_image(self, system_prompt: str, image_path: str):
        """Process a single image file (PNG, JPG, JPEG, GIF, BMP, etc.)"""
        if self._openai_client is None:
            raise ValueError("OpenAI Client non inizializzato correttamente")
        
        try:
            from PIL import Image
            from io import BytesIO
            import base64
            import os
            
            print(f"Processing image: {image_path}")
            
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Open and process image (PIL auto-detects format)
            image = Image.open(image_path)
            print(f"Image format: {image.format}, size: {image.size}, mode: {image.mode}")
            
            # Convert to RGB if necessary (handles RGBA, CMYK, grayscale, etc.)
            if image.mode not in ('RGB', 'L'):
                print(f"Converting from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Determine output format based on original (preserve JPEG as JPEG for smaller size)
            output_format = "JPEG" if image.format in ["JPEG", "JPG"] else "PNG"
            mime_type = "image/jpeg" if output_format == "JPEG" else "image/png"
            
            # Convert to base64
            buffered = BytesIO()
            if output_format == "JPEG":
                image.save(buffered, format="JPEG", quality=95, optimize=True)
            else:
                image.save(buffered, format="PNG", optimize=True)
            
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            print(f"Output format: {output_format}, Base64 length: {len(img_base64)}")
            
            # Prepare content
            content = [
                {"type": "text", "text": system_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{img_base64}",
                        "detail": "high"
                    }
                }
            ]
            
            print("Sending to OpenAI API...")
            response = self._openai_client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[{"role": "user", "content": content}]
            )
            
            # Print token usage
            if response.usage:
                print(f"\n Token Usage:")
                print(f"  Input tokens:  {response.usage.prompt_tokens}")
                print(f"  Output tokens: {response.usage.completion_tokens}")
                print(f"  Total tokens:  {response.usage.total_tokens}")
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            raise