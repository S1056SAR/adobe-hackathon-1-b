import fitz
import re
import json
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from collections import defaultdict

class MultiDocumentProcessor:
    """Enhanced document processor for handling multiple PDFs with section extraction for Round 1B."""
    
    def __init__(self):
        self.processed_documents = []
        self.document_stats = {}
    
    def process_document_collection(self, pdf_paths: List[str]) -> List[Dict]:
        """Process multiple PDFs and extract structured content with sections."""
        documents = []
        
        print(f"Processing {len(pdf_paths)} documents...")
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            try:
                print(f"  Processing document {i}/{len(pdf_paths)}: {Path(pdf_path).name}")
                doc_data = self._process_single_document(pdf_path)
                if doc_data:
                    documents.append(doc_data)
                    print(f"    Successfully extracted {len(doc_data['sections'])} sections")
                else:
                    print(f"    Failed to process document")
            except Exception as e:
                print(f"    Error processing {pdf_path}: {e}")
                continue
        
        print(f"Successfully processed {len(documents)} out of {len(pdf_paths)} documents")
        return documents
    
    def _process_single_document(self, pdf_path: str) -> Dict:
        """Process a single PDF and extract sections with rich content."""
        doc = fitz.open(pdf_path)
        filename = Path(pdf_path).name
        
        try:
            # Extract document structure and content
            sections = self._extract_sections_with_content(doc)
            full_text = self._extract_full_text(doc)
            doc_stats = self._analyze_document_statistics(doc)
            
            return {
                'filename': filename,
                'sections': sections,
                'full_text': full_text,
                'total_pages': len(doc),
                'stats': doc_stats
            }
        finally:
            doc.close()
    
    def _extract_sections_with_content(self, doc: fitz.Document) -> List[Dict]:
        """Extract sections with their content using advanced heading detection."""
        sections = []
        current_section = None
        
        # Analyze document-wide font statistics for better heading detection
        font_stats = self._analyze_font_statistics(doc)
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                # Extract block text and metadata
                block_data = self._extract_block_data(block)
                if not block_data['text'].strip():
                    continue
                
                # Determine if this is a section heading using advanced analysis
                if self._is_section_heading_advanced(block_data, font_stats, page):
                    # Save previous section if exists
                    if current_section and current_section['content'].strip():
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        'title': block_data['text'].strip(),
                        'page': page_num + 1,
                        'content': '',
                        'subsections': [],
                        'metadata': {
                            'font_size': block_data['max_font_size'],
                            'is_bold': block_data['is_bold'],
                            'is_centered': block_data['is_centered']
                        }
                    }
                else:
                    # Add content to current section
                    if current_section:
                        current_section['content'] += block_data['text'] + '\n\n'
                    else:
                        # Create a default section for content without heading
                        current_section = {
                            'title': self._generate_section_title(block_data['text']),
                            'page': page_num + 1,
                            'content': block_data['text'] + '\n\n',
                            'subsections': [],
                            'metadata': {
                                'font_size': block_data['max_font_size'],
                                'is_bold': block_data['is_bold'],
                                'is_centered': block_data['is_centered']
                            }
                        }
        
        # Add the last section
        if current_section and current_section['content'].strip():
            sections.append(current_section)
        
        # Post-process sections to clean up and merge small ones
        processed_sections = self._post_process_sections(sections)
        
        return processed_sections
    
    def _extract_block_data(self, block) -> Dict:
        """Extract comprehensive data from a text block."""
        text = ""
        font_sizes = []
        fonts = []
        is_bold = False
        is_italic = False
        bbox = block.get('bbox', [0, 0, 0, 0])
        
        for line in block["lines"]:
            for span in line["spans"]:
                span_text = span["text"].strip()
                if span_text:
                    text += span_text + " "
                    font_sizes.append(span.get('size', 12))
                    fonts.append(span.get('font', ''))
                    
                    # Check font characteristics
                    font = span.get('font', '').lower()
                    flags = span.get('flags', 0)
                    
                    if 'bold' in font or (flags & 2**4):
                        is_bold = True
                    if 'italic' in font or (flags & 2**1):
                        is_italic = True
        
        # Calculate block statistics
        max_font_size = max(font_sizes) if font_sizes else 12
        avg_font_size = np.mean(font_sizes) if font_sizes else 12
        
        # Check if text is centered (simplified check)
        is_centered = self._is_text_centered(bbox, text)
        
        return {
            'text': text.strip(),
            'max_font_size': max_font_size,
            'avg_font_size': avg_font_size,
            'is_bold': is_bold,
            'is_italic': is_italic,
            'is_centered': is_centered,
            'bbox': bbox,
            'fonts': fonts,
            'word_count': len(text.split())
        }
    
    def _analyze_font_statistics(self, doc: fitz.Document) -> Dict:
        """Analyze document-wide font statistics for better heading detection."""
        all_font_sizes = []
        all_fonts = []
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                all_font_sizes.append(span.get('size', 12))
                                all_fonts.append(span.get('font', ''))
        
        if not all_font_sizes:
            return {'mean_size': 12, 'std_size': 0, 'max_size': 12}
        
        return {
            'mean_size': np.mean(all_font_sizes),
            'std_size': np.std(all_font_sizes),
            'max_size': max(all_font_sizes),
            'min_size': min(all_font_sizes),
            'common_fonts': self._get_common_fonts(all_fonts)
        }
    
    def _get_common_fonts(self, fonts: List[str]) -> List[str]:
        """Get most common fonts in the document."""
        font_counts = defaultdict(int)
        for font in fonts:
            font_counts[font] += 1
        
        # Return top 3 most common fonts
        return sorted(font_counts.keys(), key=lambda x: font_counts[x], reverse=True)[:3]
    
    def _is_section_heading_advanced(self, block_data: Dict, font_stats: Dict, page) -> bool:
        """Advanced heading detection using multiple signals."""
        text = block_data['text']
        
        # Basic filters
        if len(text) < 3 or len(text) > 250:
            return False
        
        if block_data['word_count'] > 20:  # Headings are usually short
            return False
        
        # Pattern-based detection (FIXED: was 'patterns', now 'heading_patterns')
        heading_patterns = [
            r'^\d+\.\s+[A-Z]',  # "1. Introduction"
            r'^\d+\.\d+\s+[A-Z]',  # "1.1 Overview"
            r'^\d+\.\d+\.\d+\s+[A-Z]',  # "1.1.1 Details"
            r'^(Chapter|Section|Part|Appendix)\s+\d+',  # "Chapter 1"
            r'^(Introduction|Summary|Conclusion|Abstract|References|Discussion|Methodology|Results|Background|Overview|Objectives|Scope|Literature Review|Data Analysis|Findings)$',
            r'^[A-Z][A-Za-z\s]+:?\s*$',  # Title case
            r'^[A-Z\s]+$',  # ALL CAPS (but not too long)
        ]
        
        has_pattern = any(re.match(pattern, text, re.IGNORECASE) for pattern in heading_patterns)
        
        # Font-based analysis
        font_size_threshold = font_stats['mean_size'] + 0.5 * font_stats['std_size']
        is_large_font = block_data['max_font_size'] > font_size_threshold
        
        # Multi-factor scoring
        score = 0
        
        # Pattern matching (highest weight)
        if has_pattern:
            score += 4
        
        # Font characteristics
        if block_data['is_bold']:
            score += 3
        if is_large_font:
            score += 2
        if block_data['is_centered']:
            score += 2
        if block_data['is_italic']:
            score += 1
        
        # Text characteristics
        if text.istitle():
            score += 1
        if text.isupper() and block_data['word_count'] <= 6:
            score += 2
        if block_data['word_count'] <= 8:
            score += 1
        
        # Position-based scoring
        bbox = block_data['bbox']
        page_height = page.rect.height
        relative_y = bbox[1] / page_height if page_height > 0 else 0
        
        if relative_y < 0.15:  # Top of page
            score += 1
        
        # Negative scoring for unlikely headings
        if text.endswith('.') and block_data['word_count'] > 5:
            score -= 2
        if any(word in text.lower() for word in ['page', 'fig', 'table', 'equation']):
            score -= 3
        
        # Threshold for heading classification
        return score >= 5
    
    def _is_text_centered(self, bbox: List[float], text: str) -> bool:
        """Check if text appears to be centered (simplified approach)."""
        if not bbox or len(bbox) < 4:
            return False
        
        # This is a simplified check - in a real implementation,
        # you'd compare with page width and surrounding elements
        text_width = bbox[2] - bbox[0]
        return len(text) < 50 and text_width < 200  # Rough heuristic
    
    def _generate_section_title(self, text: str) -> str:
        """Generate a section title from the first part of content."""
        # Take first sentence or first 50 characters
        sentences = text.split('.')
        if sentences and len(sentences[0]) < 100:
            return sentences[0].strip()
        
        words = text.split()[:8]  # First 8 words
        return ' '.join(words) + ('...' if len(text.split()) > 8 else '')
    
    def _post_process_sections(self, sections: List[Dict]) -> List[Dict]:
        """Post-process sections to clean up and improve quality."""
        if not sections:
            return sections
        
        processed = []
        
        for section in sections:
            # Clean up content
            content = section['content'].strip()
            if len(content) < 50:  # Very short sections might be artifacts
                continue
            
            # Clean up title
            title = section['title'].strip()
            if not title:
                continue
            
            # Remove common artifacts from title
            title = re.sub(r'^\d+\s*$', '', title)  # Just numbers
            title = re.sub(r'^[^\w]+', '', title)   # Leading non-word chars
            
            if len(title) < 2:
                continue
            
            processed_section = {
                'title': title,
                'page': section['page'],
                'content': content,
                'subsections': section.get('subsections', []),
                'metadata': section.get('metadata', {})
            }
            
            processed.append(processed_section)
        
        return processed
    
    def _extract_full_text(self, doc: fitz.Document) -> str:
        """Extract all text from the document."""
        full_text = ""
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                full_text += page_text + "\n\n"
        return full_text.strip()
    
    def _analyze_document_statistics(self, doc: fitz.Document) -> Dict:
        """Analyze document-wide statistics."""
        total_chars = 0
        total_words = 0
        total_blocks = 0
        
        for page in doc:
            page_text = page.get_text()
            total_chars += len(page_text)
            total_words += len(page_text.split())
            
            blocks = page.get_text("dict")["blocks"]
            total_blocks += len([b for b in blocks if "lines" in b])
        
        return {
            'total_pages': len(doc),
            'total_characters': total_chars,
            'total_words': total_words,
            'total_blocks': total_blocks,
            'avg_words_per_page': total_words / max(len(doc), 1),
            'avg_chars_per_page': total_chars / max(len(doc), 1)
        }

class EnhancedSectionExtractor:
    """Enhanced section extractor with better content organization."""
    
    def __init__(self):
        self.section_keywords = [
            'introduction', 'background', 'overview', 'summary', 'conclusion',
            'methodology', 'methods', 'approach', 'results', 'findings',
            'discussion', 'analysis', 'evaluation', 'references', 'bibliography',
            'abstract', 'acknowledgments', 'appendix', 'glossary'
        ]
    
    def extract_enhanced_sections(self, doc: fitz.Document) -> List[Dict]:
        """Extract sections with enhanced content organization."""
        # This method can be used for more specialized section extraction
        # if needed for specific document types
        pass
    
    def identify_section_hierarchy(self, sections: List[Dict]) -> List[Dict]:
        """Identify hierarchical relationships between sections."""
        # Enhanced hierarchy detection based on numbering, indentation, etc.
        pass
