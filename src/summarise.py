# src/summarise.py
from llama_cpp import Llama

class Summariser:
    def __init__(self):
        print(f"   - Initializing Llama.cpp model for summarization...")
        self.llm = Llama(
            model_path="./models/tinyllama-1.1b-gguf/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf", # Point to your Q5_K_M model
            n_gpu_layers=0,
            n_ctx=2048, # Standard context window for TinyLlama
            verbose=False
        )
        print("   - Llama.cpp model initialized for summarization.")

    def refine(self, section, persona, job):
        # Even more refined prompt, strongly emphasizing word limit and paragraph format
        prompt = (
            f"You are a {persona}. Your primary goal is to {job.strip()}. "
            f"Given the following document section, summarize the most important, "
            f"actionable information strictly in **under 80 words**. "
            f"Write a single, coherent paragraph. Do NOT use bullet points or lists. "
            f"Do NOT include any introductory phrases like 'Based on the text,' or 'This section discusses.' "
            f"If the section contains no information relevant to your goal, simply respond with '[No relevant information]'.\n\n"
            f"Document: {section['doc']}\n"
            f"Page: {section['page']}\n"
            f"Section Title: {section['title']}\n"
            f"Original Text:\n{section['text']}\n\n"
            f"Concise Actionable Summary:" # Simpler ending
        )
        
        output = self.llm.create_completion(
            prompt=prompt,
            max_tokens=100, # <--- CRITICAL CHANGE: Reduced from 150 to 100
            temperature=0.3, # <--- CRITICAL CHANGE: Lowered from 0.4 to 0.3 (more deterministic)
            top_p=0.7,       # <--- CRITICAL CHANGE: Lowered from 0.8 to 0.7 (more focused)
            # Refined stop words for more aggressive truncation if LLM deviates
            stop=["Concise Actionable Summary:", "\n\n", "Original Text:", "Document:", "Page:", "Section Title:", "[No relevant information]"], 
            repeat_penalty=1.1, 
            top_k=40,
            tfs_z=0.9
        )
        
        generated_text = output["choices"][0]["text"].strip()
        
        # Aggressive post-processing
        unwanted_phrases_lower = [
            "summary:", "actionable summary:", "this section discusses:", "based on the text,",
            "this section provides information on", "in this section,", "the following text describes",
            "the provided text highlights", "this excerpt highlights", "the section outlines",
            "concise actionable summary:", # Added the new prompt ending
            "summary for travel planner (plan a trip of 4 days for a group of 10 college friends.):",
            "[no relevant information]" # Check for this as an explicit response
        ]
        
        cleaned_text = generated_text
        for phrase in unwanted_phrases_lower:
            if cleaned_text.lower().startswith(phrase):
                cleaned_text = cleaned_text[len(phrase):].strip()
        
        # Additional check to remove bullet points if TinyLlama ignores instruction
        cleaned_text = cleaned_text.replace("- ", "").replace("• ", "").strip()

        # Final check for minimal content or specific non-relevant phrases
        # Increased minimum words as well.
        if "[no relevant information]" in cleaned_text.lower() or len(cleaned_text.split()) < 20 or cleaned_text.count(' ') < 10:
            return {"doc": section["doc"], "page": section["page"], "text": ""} 

        return {"doc": section["doc"], "page": section["page"], "text": cleaned_text}

def refine_texts(ranked, persona, job):
    print("   - Initializing summarization process...")
    summariser = Summariser() 
    print("   - Summarization process ready. Starting text refinement...")
    
    refined_results = []
    for i, sec in enumerate(ranked):
        print(f"     - Refining section {i + 1}/{len(ranked)}: {sec['title']} (Page {sec['page']})...")
        refined_text_obj = summariser.refine(sec, persona, job)
        refined_results.append(refined_text_obj)
    
    return refined_results
