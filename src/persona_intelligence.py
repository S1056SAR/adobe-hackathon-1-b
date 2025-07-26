import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import Dict, List, Any
import os

class PersonaIntelligenceEngine:
    """Advanced persona-driven document analysis using Qwen2.5-0.5B-Instruct."""
    
    def __init__(self, model_path: str = "./models/qwen"):
        """Initialize the Qwen model for persona analysis."""
        print("Loading Qwen2.5-0.5B-Instruct model...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        # Set model to evaluation mode
        self.model.eval()
        print("Model loaded successfully!")
    
    def analyze_documents_for_persona(self, documents: List[Dict], persona: str, job_to_be_done: str) -> Dict:
        """Main analysis method that processes documents based on persona and job."""
        
        # Build comprehensive analysis prompt
        analysis_prompt = self._build_comprehensive_prompt(documents, persona, job_to_be_done)
        
        # Generate analysis using Qwen
        analysis_result = self._generate_analysis(analysis_prompt)
        
        # Parse and structure the output
        structured_output = self._parse_qwen_output(analysis_result, documents)
        
        return structured_output
    
    def _build_comprehensive_prompt(self, documents: List[Dict], persona: str, job_to_be_done: str) -> str:
        """Build a comprehensive prompt for Qwen analysis."""
        
        # Format documents for the prompt
        doc_summaries = []
        for i, doc in enumerate(documents, 1):
            sections_text = ""
            for j, section in enumerate(doc['sections'][:8]):  # Limit sections for prompt size
                content_preview = section['content'][:400] + "..." if len(section['content']) > 400 else section['content']
                sections_text += f"    Section: {section['title']} (Page {section['page']})\n    Content: {content_preview}\n\n"
            
            doc_summaries.append(f"Document: {doc['filename']}\n{sections_text}")
        
        documents_text = "\n".join(doc_summaries)
        
        prompt = f"""You are an expert document analyst specializing in persona-driven content extraction and ranking.

PERSONA: {persona}
JOB TO BE DONE: {job_to_be_done}

DOCUMENT COLLECTION:
{documents_text}

Your task is to analyze these documents from the perspective of the specified persona and their specific job requirements. You need to:

1. Identify the most relevant sections across all documents that would help accomplish the job
2. Rank sections by importance (1 = most important, lower numbers = higher importance)
3. Extract and refine key subsections with content specifically tailored for the persona

Please provide your analysis in the following JSON format:

{{
    "section_ranking": [
        {{
            "document": "exact_filename.pdf",
            "section_title": "exact section title from the document",
            "importance_rank": ranking_number,
            "page_number": page_number,
            "relevance_reason": "why this section is important for the persona and job"
        }}
    ],
    "refined_subsections": [
        {{
            "document": "exact_filename.pdf",
            "page_number": page_number,
            "refined_text": "detailed, refined content tailored specifically for the {persona} to accomplish: {job_to_be_done}",
            "relevance_context": "why this content is valuable"
        }}
    ]
}}

Focus on practical, actionable information that directly supports the persona in completing their specific job. Prioritize sections that provide concrete guidance, specific details, and relevant insights."""

        return prompt
    
    def _generate_analysis(self, prompt: str) -> str:
        """Generate analysis using Qwen model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3500  # Leave room for generation
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + 2000,  # Allow for response
                    temperature=0.1,  # Low temperature for consistency
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part (after the prompt)
            generated_text = response[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Error in generation: {e}")
            return self._create_fallback_analysis()
    
    def _parse_qwen_output(self, qwen_output: str, documents: List[Dict]) -> Dict:
        """Parse Qwen output and structure it according to the required format."""
        
        try:
            # Try to extract JSON from the output
            json_match = re.search(r'\{.*\}', qwen_output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_json = json.loads(json_str)
                
                # Convert to required format
                extracted_sections = []
                subsection_analysis = []
                
                # Process section ranking
                for section in parsed_json.get('section_ranking', [])[:15]:  # Limit to top 15
                    extracted_sections.append({
                        "document": section.get('document', ''),
                        "section_title": section.get('section_title', ''),
                        "importance_rank": section.get('importance_rank', 999),
                        "page_number": section.get('page_number', 1)
                    })
                
                # Process refined subsections
                for subsection in parsed_json.get('refined_subsections', [])[:10]:  # Limit to top 10
                    subsection_analysis.append({
                        "document": subsection.get('document', ''),
                        "refined_text": subsection.get('refined_text', ''),
                        "page_number": subsection.get('page_number', 1)
                    })
                
                # Sort sections by importance rank
                extracted_sections.sort(key=lambda x: x['importance_rank'])
                
                return {
                    "extracted_sections": extracted_sections,
                    "subsection_analysis": subsection_analysis  # Corrected key name
                }
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"JSON parsing error: {e}")
            
        # Fallback: create analysis from available data
        return self._create_fallback_structured_output(documents)
    
    def _create_fallback_analysis(self) -> str:
        """Create a fallback analysis when generation fails."""
        return """
        {
            "section_ranking": [
                {
                    "document": "document1.pdf",
                    "section_title": "Introduction",
                    "importance_rank": 1,
                    "page_number": 1,
                    "relevance_reason": "Provides foundational context"
                }
            ],
            "refined_subsections": [
                {
                    "document": "document1.pdf",
                    "page_number": 1,
                    "refined_text": "Key concepts and background information relevant to the analysis.",
                    "relevance_context": "Essential background"
                }
            ]
        }
        """
    
    def _create_fallback_structured_output(self, documents: List[Dict]) -> Dict:
        """Create structured output when JSON parsing fails."""
        extracted_sections = []
        subsection_analysis = []
        
        # Create basic analysis from document sections
        rank = 1
        for doc in documents[:5]:  # Process first 5 documents
            for section in doc['sections'][:3]:  # First 3 sections per doc
                extracted_sections.append({
                    "document": doc['filename'],
                    "section_title": section['title'],
                    "importance_rank": rank,
                    "page_number": section['page']
                })
                
                # Create refined text
                refined_text = section['content'][:600] + "..." if len(section['content']) > 600 else section['content']
                subsection_analysis.append({
                    "document": doc['filename'],
                    "refined_text": f"Key insights from {section['title']}: {refined_text}",
                    "page_number": section['page']
                })
                
                rank += 1
                if rank > 10:  # Limit total sections
                    break
            if rank > 10:
                break
        
        return {
            "extracted_sections": extracted_sections[:10],
            "subsection_analysis": subsection_analysis[:8]  # Corrected key name
        }
