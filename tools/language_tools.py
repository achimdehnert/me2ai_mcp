"""Language learning tools for agents."""
import json
from typing import Dict, Any
from .base import BaseTool

class GermanDictionaryTool(BaseTool):
    """Tool for looking up German words and their meanings."""
    
    def __init__(self):
        """Initialize the dictionary tool."""
        super().__init__(
            name="german_dictionary",
            description="Look up German words in a dictionary",
            parameters={
                "word": {
                    "type": "string",
                    "description": "German word to look up"
                }
            }
        )
        # Sample dictionary data (in practice, you'd use a real API)
        self.dictionary = {
            "hallo": {
                "translation": "hello",
                "type": "greeting",
                "examples": ["Hallo, wie geht's?", "Hallo zusammen!"]
            },
            "danke": {
                "translation": "thank you",
                "type": "expression",
                "examples": ["Danke schön!", "Vielen Dank!"]
            }
        }
    
    def run(self, word: str) -> Dict[str, Any]:
        """Look up a German word.
        
        Args:
            word: German word to look up
            
        Returns:
            Dict[str, Any]: Word definition and examples
        """
        word = word.lower()
        if word in self.dictionary:
            entry = self.dictionary[word]
            return {
                "word": word,
                "translation": entry['translation'],
                "type": entry['type'],
                "examples": entry['examples']
            }
        return {"error": f"Word '{word}' not found in dictionary."}

class GrammarCheckerTool(BaseTool):
    """Tool for checking German grammar."""
    
    def __init__(self):
        """Initialize the grammar checker tool."""
        super().__init__(
            name="grammar_check",
            description="Check German grammar and get corrections",
            parameters={
                "text": {
                    "type": "string",
                    "description": "German text to check"
                }
            }
        )
        
    def run(self, text: str) -> Dict[str, Any]:
        """Check German grammar.
        
        Args:
            text: German text to check
            
        Returns:
            Dict[str, Any]: Grammar check results
        """
        # In practice, you'd use a real grammar checking API
        # This is a simplified example
        common_errors = {
            "ich bin müde": None,  # Correct
            "ich bin mude": "ich bin müde",  # Missing umlaut
            "ich gehe zu hause": "ich gehe nach Hause",  # Wrong preposition
        }
        
        text = text.lower()
        if text in common_errors:
            correction = common_errors[text]
            if correction:
                return {
                    "text": text,
                    "correction": correction
                }
            return {
                "text": text,
                "result": "Text is grammatically correct."
            }
        
        return {
            "text": text,
            "error": "Unable to check grammar for this text."
        }
