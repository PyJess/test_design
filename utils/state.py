from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class TestDesignState(BaseModel):
    input_path: Optional[str] = None 
    mapping: Optional[Any] = None
    headers: Optional[List] = None
    paragraphs: Optional[List] = None
    filtered_paragraphs:Optional[List] = None
    filtered_headers: Optional[List] = None
    vectorstore: Optional[Any] = None
    updated_json: Dict[str, Any] = None
    updated_json: Optional[Dict[str, Any]] = None  
    output_json_path: Optional[str] = None  
    output_excel_path: Optional[str] = None
    docx_input_path: Optional[str] = None
    total_test_cases: Optional[int] = None
    input_dictionary: Optional[Dict] = None


    class Config:
        arbitrary_types_allowed = True