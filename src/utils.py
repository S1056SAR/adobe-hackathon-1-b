import json
import os
from datetime import datetime
from typing import Dict, List, Any

def read_input_configuration(input_dir: str) -> Dict:
    """Read the input.json configuration file."""
    input_config_path = os.path.join(input_dir, 'input.json')
    
    if not os.path.exists(input_config_path):
        raise FileNotFoundError(f"input.json not found in {input_dir}")
    
    with open(input_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def get_pdf_files_from_config(input_dir: str, config: Dict) -> List[str]:
    """Get PDF file paths based on the input configuration."""
    pdf_dir = os.path.join(input_dir, 'PDFs')
    pdf_files = []
    
    # Get filenames from config
    document_filenames = [doc['filename'] for doc in config['documents']]
    
    for filename in document_filenames:
        pdf_path = os.path.join(pdf_dir, filename)
        if os.path.exists(pdf_path):
            pdf_files.append(pdf_path)
        else:
            print(f"Warning: PDF file not found: {pdf_path}")
    
    return pdf_files

def create_metadata_from_config(config: Dict) -> Dict:
    """Create metadata from input configuration."""
    return {
        "input_documents": [doc['filename'] for doc in config['documents']],
        "persona": config['persona']['role'],
        "job_to_be_done": config['job_to_be_done']['task'],
        "processing_timestamp": datetime.now().isoformat()
    }

def validate_output_format(output_data: Dict) -> bool:
    """Validate that the output matches the required format."""
    required_keys = ['metadata', 'extracted_sections', 'subsection_analysis']
    
    # Check main structure
    if not all(key in output_data for key in required_keys):
        return False
    
    # Validate metadata
    metadata_keys = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']
    if not all(key in output_data['metadata'] for key in metadata_keys):
        return False
    
    # Validate extracted_sections
    for section in output_data['extracted_sections']:
        section_keys = ['document', 'section_title', 'importance_rank', 'page_number']
        if not all(key in section for key in section_keys):
            return False
    
    # Validate subsection_analysis (note: corrected key name)
    for subsection in output_data['subsection_analysis']:
        subsection_keys = ['document', 'refined_text', 'page_number']
        if not all(key in subsection for key in subsection_keys):
            return False
    
    return True

def save_json_output(data: Dict, output_path: str) -> None:
    """Save data as JSON file with proper formatting."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
