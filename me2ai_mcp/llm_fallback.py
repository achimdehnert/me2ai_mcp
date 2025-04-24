"""
LLM Fallback Mixin for ME2AI MCP agents.

This module provides a standardized mixin for implementing robust multi-LLM fallback
patterns in agents, prioritizing OpenAI, then Anthropic, and finally falling back
to heuristic methods if no LLMs are available.
"""
from typing import Dict, List, Any, Optional, Union, Callable, Type, Tuple
import logging
import os
import json
import re
from enum import Enum, auto


class LLMProvider(Enum):
    """Enum representing available LLM providers in priority order."""
    OPENAI = auto()
    ANTHROPIC = auto()
    HEURISTIC = auto()


class LLMFallbackMixin:
    """
    Mixin providing standardized LLM fallback functionality for ME2AI MCP agents.
    
    This mixin implements the robust multi-LLM fallback pattern:
    1. First attempts to use OpenAI
    2. Falls back to Anthropic if OpenAI is unavailable
    3. Falls back to heuristic methods if both LLMs are unavailable
    
    The mixin provides common utilities for LLM-based tasks such as:
    - Text summarization
    - Entity extraction
    - Question generation
    - Embedding generation
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the LLMFallbackMixin."""
        # Set up logging if not already configured
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Call the parent class's __init__ if applicable
        super().__init__(*args, **kwargs)
    
    def get_available_llm(self) -> Optional[Tuple[LLMProvider, Any]]:
        """
        Get the first available LLM from the fallback chain.
        
        Returns:
            A tuple with LLM provider enum and LLM object, or None if no LLMs available
        """
        # Try OpenAI (primary)
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain.llms import OpenAI
                self.logger.info("Using OpenAI as primary LLM provider")
                return LLMProvider.OPENAI, OpenAI(temperature=0)
            except (ImportError, Exception) as e:
                self.logger.warning(f"Failed to initialize OpenAI: {str(e)}")
        else:
            self.logger.info("OpenAI API key not set, checking backup LLM")
        
        # Try Anthropic (backup)
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                from langchain.llms import Anthropic
                self.logger.info("Using Anthropic as backup LLM provider")
                return LLMProvider.ANTHROPIC, Anthropic(temperature=0)
            except (ImportError, Exception) as e:
                self.logger.warning(f"Failed to initialize Anthropic: {str(e)}")
        else:
            self.logger.info("Anthropic API key not set")
        
        # No LLMs available
        self.logger.warning("No LLM providers available, falling back to heuristic methods")
        return None
    
    def run_llm_chain(self, 
                      prompt_template: str, 
                      content: str = "", 
                      max_retries: int = 2,
                      logger: Optional[logging.Logger] = None) -> Optional[str]:
        """
        Run a prompt through the LLM fallback chain.
        
        Args:
            prompt_template: The prompt template to use
            content: Optional content to format into the template
            max_retries: Maximum number of retries per LLM provider
            logger: Optional logger to use (defaults to self.logger)
        
        Returns:
            The LLM response or None if all LLMs fail
        """
        log = logger or self.logger
        
        # Try to get an available LLM
        llm_result = self.get_available_llm()
        if not llm_result:
            return None
        
        provider, llm = llm_result
        
        # Format the prompt if needed (support both {} format and {text} template)
        formatted_prompt = prompt_template
        if "{text}" in prompt_template and content:
            formatted_prompt = prompt_template.format(text=content)
        elif content and "{}" in prompt_template:
            formatted_prompt = prompt_template.format(content)
        
        # Try the selected LLM
        for attempt in range(max_retries):
            try:
                if provider == LLMProvider.OPENAI or provider == LLMProvider.ANTHROPIC:
                    # Run the LLM
                    response = llm(formatted_prompt)
                    return response
            except Exception as e:
                log.warning(f"LLM call failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    log.error(f"All attempts failed with LLM provider {provider.name}")
        
        # If we get here, all attempts with the primary LLM failed
        # Try the fallback LLM if not already tried
        if provider == LLMProvider.OPENAI and os.getenv("ANTHROPIC_API_KEY"):
            log.info("Primary LLM (OpenAI) failed, trying backup (Anthropic)")
            try:
                from langchain.llms import Anthropic
                anthropic_llm = Anthropic(temperature=0)
                response = anthropic_llm(formatted_prompt)
                return response
            except Exception as e:
                log.error(f"Backup LLM (Anthropic) failed: {str(e)}")
        
        # If we get here, all LLMs failed
        log.warning("All LLM providers failed, falling back to heuristic methods")
        return None
    
    def extract_entities_with_fallback(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text with LLM fallback.
        
        Args:
            text: Text to extract entities from
        
        Returns:
            List of entity dictionaries with name, type, and relevance
        """
        # First try with LLM
        prompt = """
        Extract entities from the following text. Return the result as JSON in the format:
        [
            {"name": "Entity Name", "type": "PERSON/ORGANIZATION/LOCATION/PRODUCT/EVENT/DATE/OTHER", "relevance": 0.9}
        ]
        Text: {text}
        
        JSON Entities:
        """
        
        result = self.run_llm_chain(prompt, text)
        
        if result:
            try:
                entities = json.loads(result)
                if isinstance(entities, list):
                    return entities
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse LLM entity extraction output as JSON")
        
        # Fallback to regex-based extraction
        self.logger.info("Using regex fallback for entity extraction")
        entities = []
        
        # Simple patterns for common entity types
        patterns = {
            "PERSON": r"([A-Z][a-z]+ [A-Z][a-z]+)",
            "ORGANIZATION": r"([A-Z][a-zA-Z]*([ \-&](?:[A-Z][a-zA-Z]*|[Tt]he|[Oo]f|[Aa]nd|[Ii]nc|[Ll]td))+)",
            "LOCATION": r"([A-Z][a-z]+ (?:City|Island|Mountain|Street|Road|Avenue|Boulevard|County|State|Country))",
            "DATE": r"(\d{1,2}\/\d{1,2}\/\d{2,4}|\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{2,4})",
            "URL": r"(https?:\/\/[^\s]+)",
            "EMAIL": r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                # Skip duplicates
                if not any(e["name"] == match for e in entities):
                    entities.append({
                        "name": match,
                        "type": entity_type,
                        "relevance": 0.7,  # Lower confidence for regex-based extraction
                    })
        
        return entities
    
    def summarize_text_with_fallback(self, text: str, max_length: int = 500) -> Dict[str, Any]:
        """
        Summarize text with LLM fallback.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
        
        Returns:
            Dictionary with summary and metadata
        """
        # First try with LLM
        prompt = """
        Please create a concise summary of the following text. The summary should capture
        the main points and key information while being significantly shorter than the original.
        
        Text to summarize: {text}
        
        Summary:
        """
        
        result = self.run_llm_chain(prompt, text)
        
        if result:
            summary = result.strip()
            
            # Ensure summary isn't longer than requested maximum
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
                
            return {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "reduction_ratio": len(summary) / len(text),
                "method": "llm",
            }
        
        # Fallback to extractive summarization
        self.logger.info("Using extractive fallback for text summarization")
        
        # Split text into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        
        # Score sentences based on position, length, keywords
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Position: Prefer sentences at the beginning and end
            if i < 3:  # First 3 sentences
                score += 5 - i
            elif i >= len(sentences) - 3:  # Last 3 sentences
                score += 5 - (len(sentences) - i)
            
            # Length: Prefer medium-length sentences
            length_score = min(len(sentence) / 20, 3)
            score += length_score
            
            # Keywords: Prefer sentences with important words
            keywords = ["important", "significant", "key", "main", "crucial", "essential"]
            for keyword in keywords:
                if keyword.lower() in sentence.lower():
                    score += 2
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and select top sentences
        sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
        
        # Select sentences up to max_length
        total_length = 0
        selected_sentences = []
        
        for sentence, _ in sorted_sentences:
            if total_length + len(sentence) <= max_length:
                selected_sentences.append(sentence)
                total_length += len(sentence)
            else:
                break
        
        # Get original indices to preserve order
        original_indices = []
        for selected in selected_sentences:
            for i, sentence in enumerate(sentences):
                if selected == sentence:
                    original_indices.append(i)
                    break
        
        # Sort by original position
        ordered_summary = [selected_sentences[original_indices.index(i)] 
                          for i in sorted(original_indices)]
        
        summary = " ".join(ordered_summary)
        
        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "reduction_ratio": len(summary) / len(text),
            "method": "extractive",
        }
    
    def generate_embeddings_with_fallback(self, 
                                          texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings with LLM fallback.
        
        Args:
            texts: List of texts to generate embeddings for
        
        Returns:
            List of embedding vectors or None if all embedding methods fail
        """
        # Try OpenAI embeddings
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain.embeddings import OpenAIEmbeddings
                
                embeddings = OpenAIEmbeddings()
                return embeddings.embed_documents(texts)
            except Exception as e:
                self.logger.warning(f"OpenAI embeddings failed: {str(e)}")
        
        # Try SentenceTransformers as fallback
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(texts, convert_to_tensor=False).tolist()
        except Exception as e:
            self.logger.warning(f"SentenceTransformers failed: {str(e)}")
        
        # Use simple numpy-based fallback
        try:
            import numpy as np
            
            # Create deterministic but unique embeddings based on word frequency
            embeddings = []
            
            for text in texts:
                # Simple word frequency vector
                words = text.lower().split()
                unique_words = list(set(words))
                
                # Create a pseudo-random but deterministic embedding
                embedding = []
                for i in range(384):  # Standard small embedding size
                    # Hash-based approach to create a deterministic vector
                    value = sum(hash(word + str(i)) % 10000 for word in unique_words)
                    value = (value / 10000 * 2) - 1  # Scale to -1 to 1
                    embedding.append(float(value))
                
                # Normalize the embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = [x / norm for x in embedding]
                
                embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            self.logger.error(f"All embedding methods failed: {str(e)}")
            return None
