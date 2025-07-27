import os
import time
from document_processor import MultiDocumentProcessor
from persona_intelligence import HybridPersonaIntelligence as PersonaIntelligenceEngine
from utils import (
    validate_output_format, save_json_output, read_input_configuration,
    get_pdf_files_from_config, create_metadata_from_config
)

def main():
    """Main execution function for Round 1B with corrected input/output format."""
    print("Adobe Hackathon Round 1B - Persona-Driven Document Intelligence")
    print("Using Qwen2.5-0.5B-Instruct for Advanced Analysis")
    print("=" * 70)
    
    # Setup directories
    input_dir = "./input"
    output_dir = "./output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Read input configuration
        print("Reading input configuration...")
        config = read_input_configuration(input_dir)
        
        # Extract persona and job information
        persona = config['persona']['role']
        job_to_be_done = config['job_to_be_done']['task']
        
        print(f" Persona: {persona}")
        print(f"Job: {job_to_be_done}")
        
        # Get PDF files based on configuration
        pdf_files = get_pdf_files_from_config(input_dir, config)
        
        if not pdf_files:
            print("No PDF files found based on configuration")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        print("=" * 70)
        
        start_time = time.time()
        
        # Initialize processors
        print("Initializing document processor...")
        doc_processor = MultiDocumentProcessor()
        
        print("Initializing persona intelligence engine...")
        persona_engine = PersonaIntelligenceEngine()
        
        # Process documents
        print("Processing document collection...")
        documents = doc_processor.process_document_collection(pdf_files)
        
        if not documents:
            print(" No documents could be processed successfully")
            return
        
        print(f" Successfully processed {len(documents)} documents")
        
        # Analyze with persona intelligence
        print("Performing persona-driven analysis...")
        analysis_result = persona_engine.analyze_documents_for_persona(
            documents, persona, job_to_be_done
        )
        
        # Create final output structure with correct format
        metadata = create_metadata_from_config(config)
        
        final_output = {
            "metadata": metadata,
            "extracted_sections": analysis_result.get("extracted_sections", []),
            "subsection_analysis": analysis_result.get("subsection_analysis", [])  # Corrected key
        }
        
        # Validate output format
        if not validate_output_format(final_output):
            print("Output format validation failed - but proceeding with current structure")
        
        # Save output
        output_path = os.path.join(output_dir, "analysis_result.json")
        save_json_output(final_output, output_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Print results
        print("=" * 70)
        print("Analysis Complete!")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print(f"Extracted Sections: {len(final_output['extracted_sections'])}")
        print(f"Subsection Analysis: {len(final_output['subsection_analysis'])}")
        print(f"Output saved to: {output_path}")
        
        # Show top sections
        print("\n🏆 Top Ranked Sections:")
        for i, section in enumerate(final_output['extracted_sections'][:5], 1):
            print(f"  {i}. {section.get('section_title', 'N/A')} "
                  f"(Page {section.get('page_number', 'N/A')}) "
                  f"- {section.get('document', 'N/A')} "
                  f"[Rank: {section.get('importance_rank', 'N/A')}]")
        
        print("=" * 70)
        print("Round 1B Processing Complete - Ready for Victory! 🏆")
        
    except FileNotFoundError as e:
        print(f"Input configuration error: {e}")
        print("Please ensure input.json exists in the input directory")
    except Exception as e:
        print(f"Critical error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
