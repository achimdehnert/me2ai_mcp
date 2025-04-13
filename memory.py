"""Memory management for agents."""
from typing import Any, Dict, List, Optional
from langchain_core.memory import BaseMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel, Field
from datetime import datetime

class ConversationSummary(BaseModel):
    """Summary of a conversation segment."""
    start_time: datetime
    end_time: datetime
    topic: str
    key_points: List[str]
    sentiment: str
    action_items: List[str]

class ConversationMemory(BaseMemory, BaseModel):
    """Enhanced memory class for storing conversation history with summaries."""
    
    chat_history: ChatMessageHistory = Field(default_factory=ChatMessageHistory)
    return_messages: bool = Field(default=True)
    memory_key: str = Field(default="chat_history")
    max_tokens: int = Field(default=4000)
    summaries: List[ConversationSummary] = Field(default_factory=list)
    current_topic: str = Field(default="")
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables with optional filtering."""
        messages = self.chat_history.messages
        
        if len(messages) > self.max_tokens:
            # Keep recent messages and summaries of older ones
            recent_messages = messages[-self.max_tokens:]
            return {
                self.memory_key: recent_messages,
                "summaries": self.summaries
            }
        
        return {
            self.memory_key: messages,
            "summaries": self.summaries
        }
    
    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        """Save context with enhanced tracking."""
        # Add messages to history
        self.chat_history.add_user_message(inputs["input"])
        self.chat_history.add_ai_message(outputs["output"])
        
        # Update current topic if it changes significantly
        self._update_topic(inputs["input"])
        
        # Create a summary if enough context is gathered
        if len(self.chat_history.messages) >= 10:
            self._create_summary()
    
    def clear(self) -> None:
        """Clear all memory contents."""
        self.chat_history.clear()
        self.summaries.clear()
        self.current_topic = ""
    
    def _update_topic(self, message: str) -> None:
        """Update the current conversation topic."""
        # This would typically use the LLM to analyze the message and update the topic
        # For now, we'll just store the first few words
        words = message.split()[:3]
        self.current_topic = " ".join(words) + "..."
    
    def _create_summary(self) -> None:
        """Create a summary of the recent conversation."""
        # This would typically use the LLM to generate a proper summary
        # For now, we'll create a simple one
        messages = self.chat_history.messages[-10:]
        
        summary = ConversationSummary(
            start_time=datetime.now(),  # Should be from first message
            end_time=datetime.now(),
            topic=self.current_topic,
            key_points=["Discussion about " + self.current_topic],
            sentiment="neutral",
            action_items=[]
        )
        
        self.summaries.append(summary)
        self.chat_history.messages = self.chat_history.messages[-5:]  # Keep only recent messages
    
    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables."""
        return [self.memory_key]
