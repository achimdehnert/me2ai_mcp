"""
Collaborative agent capabilities for ME2AI MCP.

This module extends the agent system with collaborative capabilities
that allow agents to work together and share context.
"""
from typing import Dict, List, Any, Optional, Union, Callable, Type, Set, Tuple
import logging
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from .agents import BaseAgent, SpecializedAgent, ToolCategory
from .base import ME2AIMCPServer, BaseTool
from .utils import sanitize_input, format_response


# Configure logging
logger = logging.getLogger("me2ai-mcp-collaborative")


@dataclass
class CollaborationContext:
    """Context for a collaboration between agents."""
    
    context_id: str
    initiator_id: str
    participants: Set[str] = field(default_factory=set)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_message(
        self, sender_id: str, message: str, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to the collaboration context.
        
        Args:
            sender_id: ID of the sending agent
            message: Message text
            data: Additional data
        """
        self.messages.append({
            "id": str(uuid.uuid4()),
            "sender_id": sender_id,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "data": data or {}
        })
        self.updated_at = datetime.now()
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the message history.
        
        Args:
            limit: Maximum number of messages to return (most recent)
            
        Returns:
            List of messages
        """
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "context_id": self.context_id,
            "initiator_id": self.initiator_id,
            "participants": list(self.participants),
            "shared_data": self.shared_data,
            "messages": self.messages,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class CollaborationManager:
    """Manager for agent collaborations."""
    
    def __init__(self) -> None:
        """Initialize the collaboration manager."""
        self.contexts: Dict[str, CollaborationContext] = {}
        self.agent_contexts: Dict[str, List[str]] = {}  # agent_id -> list of context_ids
        self.logger = logging.getLogger("me2ai-mcp-collab-manager")
    
    def create_context(
        self, initiator_id: str, participants: List[str] = None
    ) -> CollaborationContext:
        """
        Create a new collaboration context.
        
        Args:
            initiator_id: ID of the initiating agent
            participants: IDs of participating agents
            
        Returns:
            The created context
        """
        context_id = str(uuid.uuid4())
        
        # Create the context
        context = CollaborationContext(
            context_id=context_id,
            initiator_id=initiator_id
        )
        
        # Add participants
        context.participants.add(initiator_id)
        if participants:
            for participant_id in participants:
                context.participants.add(participant_id)
        
        # Store the context
        self.contexts[context_id] = context
        
        # Update agent_contexts
        for agent_id in context.participants:
            if agent_id not in self.agent_contexts:
                self.agent_contexts[agent_id] = []
            self.agent_contexts[agent_id].append(context_id)
        
        self.logger.info(
            f"Created collaboration context {context_id} for {initiator_id} "
            f"with {len(context.participants) - 1} participants"
        )
        
        return context
    
    def get_context(self, context_id: str) -> Optional[CollaborationContext]:
        """
        Get a collaboration context by ID.
        
        Args:
            context_id: Context ID
            
        Returns:
            The context or None if not found
        """
        return self.contexts.get(context_id)
    
    def get_contexts_for_agent(self, agent_id: str) -> List[CollaborationContext]:
        """
        Get all contexts an agent is participating in.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of contexts
        """
        context_ids = self.agent_contexts.get(agent_id, [])
        return [self.contexts[context_id] for context_id in context_ids if context_id in self.contexts]
    
    def add_message(
        self, context_id: str, sender_id: str, message: str, data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to a collaboration context.
        
        Args:
            context_id: Context ID
            sender_id: ID of the sending agent
            message: Message text
            data: Additional data
            
        Returns:
            Whether the message was added successfully
        """
        context = self.get_context(context_id)
        if not context:
            self.logger.warning(f"Attempted to add message to non-existent context {context_id}")
            return False
        
        if sender_id not in context.participants:
            self.logger.warning(
                f"Agent {sender_id} attempted to add message to context {context_id} "
                f"but is not a participant"
            )
            return False
        
        context.add_message(sender_id, message, data)
        self.logger.debug(f"Added message from {sender_id} to context {context_id}")
        
        return True
    
    def close_context(self, context_id: str) -> bool:
        """
        Close a collaboration context.
        
        Args:
            context_id: Context ID
            
        Returns:
            Whether the context was closed successfully
        """
        if context_id not in self.contexts:
            self.logger.warning(f"Attempted to close non-existent context {context_id}")
            return False
        
        context = self.contexts[context_id]
        
        # Remove from agent_contexts
        for agent_id in context.participants:
            if agent_id in self.agent_contexts and context_id in self.agent_contexts[agent_id]:
                self.agent_contexts[agent_id].remove(context_id)
        
        # Remove the context
        del self.contexts[context_id]
        
        self.logger.info(f"Closed collaboration context {context_id}")
        
        return True


class CollaborativeAgent(SpecializedAgent):
    """Agent capable of collaborating with other agents."""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str = "",
        server: Optional[ME2AIMCPServer] = None,
        tool_names: Optional[List[str]] = None,
        collaboration_manager: Optional[CollaborationManager] = None
    ) -> None:
        """
        Initialize a collaborative agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            description: Description of the agent's capabilities
            server: MCP server to use (optional, can be set later)
            tool_names: Specific tools this agent uses
            collaboration_manager: Manager for collaborations
        """
        super().__init__(agent_id, name, description, server, tool_names)
        
        self.collaboration_manager = collaboration_manager or global_collaboration_manager
        
        # Register collaboration tools
        if hasattr(self, "tools"):
            self.tools["request_collaboration"] = self.request_collaboration
            self.tools["send_collaboration_message"] = self.send_collaboration_message
            self.tools["get_collaboration_history"] = self.get_collaboration_history
    
    def request_collaboration(
        self, 
        target_agent_id: str, 
        request: str, 
        data: Optional[Dict[str, Any]] = None,
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Request collaboration from another agent.
        
        Args:
            target_agent_id: ID of the target agent
            request: Request text
            data: Additional data to send
            context_id: Existing context ID (optional)
            
        Returns:
            Response dictionary
        """
        if not self.server:
            return {
                "success": False,
                "error": "Agent must be attached to a server to collaborate"
            }
        
        # Get the target agent
        target_agent = self.server.get_agent(target_agent_id)
        if not target_agent:
            return {
                "success": False,
                "error": f"Agent '{target_agent_id}' not found"
            }
        
        # Get or create collaboration context
        if context_id:
            context = self.collaboration_manager.get_context(context_id)
            if not context:
                return {
                    "success": False,
                    "error": f"Collaboration context '{context_id}' not found"
                }
            if self.agent_id not in context.participants:
                context.participants.add(self.agent_id)
            if target_agent_id not in context.participants:
                context.participants.add(target_agent_id)
        else:
            context = self.collaboration_manager.create_context(
                initiator_id=self.agent_id,
                participants=[target_agent_id]
            )
        
        # Add request message
        self.collaboration_manager.add_message(
            context_id=context.context_id,
            sender_id=self.agent_id,
            message=request,
            data=data
        )
        
        # Create the collaboration request
        collaboration_request = {
            "type": "collaboration",
            "source_agent_id": self.agent_id,
            "context_id": context.context_id,
            "request": request,
            "data": data or {}
        }
        
        # Process the request with the target agent
        if hasattr(target_agent, "process_collaboration_request"):
            response = target_agent.process_collaboration_request(collaboration_request)
        else:
            # Fallback for non-collaborative agents
            response = target_agent.process_request(
                f"[Collaboration request from {self.name}] {request}"
            )
        
        # Add response to the context
        if response.get("success", False):
            self.collaboration_manager.add_message(
                context_id=context.context_id,
                sender_id=target_agent_id,
                message=response.get("message", ""),
                data=response.get("data")
            )
        
        # Add context ID to the response
        response["context_id"] = context.context_id
        
        return response
    
    def process_collaboration_request(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a collaboration request from another agent.
        
        Args:
            request: Collaboration request dictionary
            
        Returns:
            Response dictionary
        """
        # Extract request info
        source_agent_id = request.get("source_agent_id")
        context_id = request.get("context_id")
        message = request.get("request", "")
        data = request.get("data", {})
        
        self.logger.info(f"Received collaboration request from {source_agent_id}: {message[:50]}...")
        
        # Process the request (by default, use process_request)
        # Subclasses can override for special collaboration handling
        response = self.process_request(
            f"[Collaboration from {source_agent_id}] {message}"
        )
        
        # Add collaboration metadata
        response["context_id"] = context_id
        
        return response
    
    def send_collaboration_message(
        self, 
        context_id: str, 
        message: str, 
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a message to a collaboration context.
        
        Args:
            context_id: Context ID
            message: Message text
            data: Additional data
            
        Returns:
            Response dictionary
        """
        # Get the context
        context = self.collaboration_manager.get_context(context_id)
        if not context:
            return {
                "success": False,
                "error": f"Collaboration context '{context_id}' not found"
            }
        
        # Check if the agent is a participant
        if self.agent_id not in context.participants:
            return {
                "success": False,
                "error": f"Agent '{self.agent_id}' is not a participant in context '{context_id}'"
            }
        
        # Add the message
        added = self.collaboration_manager.add_message(
            context_id=context_id,
            sender_id=self.agent_id,
            message=message,
            data=data
        )
        
        if not added:
            return {
                "success": False,
                "error": f"Failed to add message to context '{context_id}'"
            }
        
        return {
            "success": True,
            "context_id": context_id,
            "message": "Message sent successfully"
        }
    
    def get_collaboration_history(
        self, context_id: str, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get the message history for a collaboration context.
        
        Args:
            context_id: Context ID
            limit: Maximum number of messages to return
            
        Returns:
            Response dictionary with message history
        """
        # Get the context
        context = self.collaboration_manager.get_context(context_id)
        if not context:
            return {
                "success": False,
                "error": f"Collaboration context '{context_id}' not found"
            }
        
        # Check if the agent is a participant
        if self.agent_id not in context.participants:
            return {
                "success": False,
                "error": f"Agent '{self.agent_id}' is not a participant in context '{context_id}'"
            }
        
        # Get the history
        history = context.get_history(limit)
        
        return {
            "success": True,
            "context_id": context_id,
            "messages": history,
            "participants": list(context.participants)
        }


# Global instance for convenience
global_collaboration_manager = CollaborationManager()
