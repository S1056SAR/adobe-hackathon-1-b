import torch
from transformers import AutoTokenizer, AutoModel
import json
import re
import os
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity

class HybridPersonaIntelligence:
    """Truly generic hybrid system with NO hardcoded document-specific rules."""
    
    def __init__(self, model_path: str = "/app/models/distilbert"):
        """Initialize generic hybrid system."""
        print("🚀 Initializing Generic Persona Intelligence System")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            
            self.model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True
            )
            
            self.model.eval()
            print("✅ DistilBERT loaded successfully!")
            
        except Exception as e:
            print(f"Error loading DistilBERT: {e}")
            raise RuntimeError(f"Failed to load DistilBERT: {e}")
    
    def analyze_documents_for_persona(self, documents: List[Dict], persona: str, job_to_be_done: str) -> Dict:
        """Generic analysis that works for ANY persona and ANY documents."""
        
        print(f"🎯 Generic analysis for {persona}: {job_to_be_done}")
        
        # Extract keywords from persona and job (generic approach)
        persona_keywords = self._extract_keywords(persona)
        job_keywords = self._extract_keywords(job_to_be_done)
        all_keywords = persona_keywords + job_keywords
        
        # Generic multi-stage analysis
        stage1_sections = self._generic_content_scoring(documents, all_keywords, persona, job_to_be_done)
        stage2_sections = self._generic_semantic_ranking(stage1_sections, persona, job_to_be_done)
        
        # Generate generic refined subsections
        subsection_analysis = self._generic_subsection_analysis(stage2_sections, persona, job_to_be_done)
        
        return {
            "extracted_sections": stage2_sections[:15],
            "subsection_analysis": subsection_analysis[:10]
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Generic keyword extraction from any text."""
        # Simple but effective keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'their', 'said', 'each', 'which', 'what', 'where', 'when', 'would', 'there', 'could', 'other', 'after', 'first', 'well', 'also', 'some', 'only', 'know', 'think', 'just', 'more', 'very', 'good', 'much', 'make', 'take', 'than', 'many', 'most', 'over', 'such', 'even', 'here', 'work', 'life', 'time', 'year', 'years', 'people', 'world', 'should', 'being', 'through', 'these', 'those', 'still', 'right', 'under', 'while', 'never', 'again', 'something', 'everything'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Return unique keywords, limited to most important ones
        return list(set(keywords))[:20]
    
    def _generic_content_scoring(self, documents: List[Dict], keywords: List[str], persona: str, job: str) -> List[Dict]:
        """Generic content scoring based on keyword relevance."""
        
        all_sections = []
        
        for doc in documents:
            for section in doc['sections']:
                title = section['title'].lower()
                content = section['content'].lower()
                
                # Generic scoring based on keyword matching
                score = 0
                
                # Keyword relevance (generic approach)
                for keyword in keywords:
                    if keyword in title:
                        score += 3  # Title matches are important
                    score += content.count(keyword) * 0.5
                
                # Content quality indicators (generic)
                content_length = len(section['content'])
                if content_length > 200:
                    score += 1
                if content_length > 500:
                    score += 1
                if content_length > 1000:
                    score += 1
                
                # Section title quality (generic patterns)
                if any(pattern in title for pattern in ['introduction', 'overview', 'summary', 'conclusion', 'abstract', 'methodology', 'analysis', 'results', 'discussion']):
                    score += 2
                
                all_sections.append({
                    "document": doc['filename'],
                    "section_title": section['title'],
                    "page_number": section['page'],
                    "content": section['content'],
                    "content_score": score
                })
        
        # Sort by content score
        all_sections.sort(key=lambda x: -x['content_score'])
        return all_sections[:25]  # Top 25 for semantic analysis
    
    def _generic_semantic_ranking(self, sections: List[Dict], persona: str, job: str) -> List[Dict]:
        """Generic semantic similarity ranking."""
        
        try:
            # Create generic query from persona and job
            query_text = f"{persona} working on {job}"
            query_embedding = self._get_embedding(query_text)
            
            # Get embeddings for all sections
            for section in sections:
                section_text = f"{section['section_title']} {section['content'][:400]}"
                section_embedding = self._get_embedding(section_text)
                
                # Calculate semantic similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    section_embedding.reshape(1, -1)
                )[0][0]
                
                # Combine content score and semantic similarity
                final_score = (0.4 * section['content_score']) + (0.6 * similarity * 100)
                section['final_score'] = final_score
            
            print("✅ Generic semantic ranking completed")
            
        except Exception as e:
            print(f"⚠️ Semantic ranking failed, using content scores: {e}")
            for section in sections:
                section['final_score'] = section['content_score']
        
        # Final ranking
        sections.sort(key=lambda x: -x['final_score'])
        
        # Assign importance ranks and clean up
        for i, section in enumerate(sections, 1):
            section['importance_rank'] = i
            del section['content_score']
            del section['final_score']
        
        return sections
    
    def _generic_subsection_analysis(self, sections: List[Dict], persona: str, job: str) -> List[Dict]:
        """Generic subsection analysis without hardcoded rules."""
        
        subsections = []
        
        for section in sections[:8]:  # Top 8 sections
            # Generic text refinement based on content analysis
            content = section['content']
            
            # Generic summarization approach
            sentences = content.split('.')[:3]  # First 3 sentences
            summary = '. '.join(sentences).strip()
            
            if len(summary) < 100:  # If too short, add more content
                summary = content[:300] + "..." if len(content) > 300 else content
            
            # Generic refinement that works for any persona/job
            refined_text = f"Relevant to your role as {persona} and task '{job}': {summary}"
            
            # Ensure reasonable length
            if len(refined_text) > 500:
                refined_text = refined_text[:497] + "..."
            
            subsections.append({
                "document": section['document'],
                "refined_text": refined_text,
                "page_number": section['page_number']
            })
            
            # Clean up section content
            del section['content']
        
        return subsections
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get DistilBERT embedding for text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        return embedding
