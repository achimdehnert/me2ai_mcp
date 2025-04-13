"""Main CLI interface for the AI coaching system."""
import os
import sys
import cmd
import asyncio
from typing import Dict, Optional, List
from dotenv import load_dotenv
from agents.factory import AgentFactory
from agents.base import BaseAgent
from llms.base import LLMProvider
from llms.openai_provider import OpenAIProvider
from llms.groq_provider import GroqProvider
from llms.anthropic_provider import AnthropicProvider

# Load environment variables
load_dotenv()

class AgentCLI(cmd.Cmd):
    """Interactive CLI for the AI coaching system."""
    
    intro = """
Welcome to the AI Expert System
---------------------------------------
Available commands:
- talk <message>: Send a message to the current agent
- switch <agent>: Switch to a different agent
  Available agents: german_professor, dating_expert, seo_expert, router, relationship_team, language_team, business_team
- auto <message>: Let the router automatically select the best expert
- list: List available agents
- clear: Clear conversation history
- help: Show this help message
- quit: Exit the program
- metrics: Show performance metrics for the current team agent
---------------------------------------
"""
    prompt = 'You> '
    
    def __init__(self, loop=None):
        """Initialize the CLI with available agents."""
        super().__init__()
        self.loop = loop or asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Set up LLM provider
        if os.getenv("GROQ_API_KEY"):
            self.llm_provider = GroqProvider()
            print("Using Groq backend with Mixtral-8x7b model")
        elif os.getenv("ANTHROPIC_API_KEY"):
            self.llm_provider = AnthropicProvider()
            print("Using Anthropic backend with Claude model")
        else:
            self.llm_provider = OpenAIProvider()
            print(f"Using OpenAI backend with {os.getenv('OPENAI_MODEL_NAME', 'gpt-4')} model")
        
        # Create agents
        self.agents = {}
        self.current_agent = 'router'
        self.auto_mode = False
    
    async def initialize(self):
        """Initialize all agents asynchronously."""
        # Create individual expert agents
        factory = AgentFactory(self.llm_provider)
        experts = {
            "german_professor": await factory.create_expert_agent("german_professor"),
            "dating_expert": await factory.create_expert_agent("dating_expert"),
            "seo_expert": await factory.create_expert_agent("seo_expert")
        }
        
        self.agents = {
            "router": await factory.create_router_agent(experts)
        }
        self.agents.update(experts)
    
    def do_talk(self, message: str) -> None:
        """Send a message to the current agent."""
        if not message:
            print("Please provide a message.")
            return
            
        print("\nProcessing...")
        agent = self.agents[self.current_agent]
        response = self.loop.run_until_complete(agent.respond(message))
        print(f"\n{agent.role}: {response}")
    
    def do_auto(self, message: str) -> None:
        """Let the router automatically select and use the best expert."""
        if not message:
            print("Please provide a message.")
            return
            
        print("\nRouting your question...")
        self.loop.run_until_complete(self._auto_route(message))
    
    async def _auto_route(self, message: str) -> None:
        """Internal async method for auto-routing."""
        router = self.agents['router']
        expert, reason = await router.get_agent(message)
        print(f"\nSelected expert: {expert.role} ({reason})")
        
        response = await expert.respond(message)
        print(f"\n{expert.role}: {response}")
    
    def do_switch(self, agent_name: str) -> None:
        """Switch to a different agent."""
        if not agent_name:
            print("Please specify an agent name.")
            return
            
        if agent_name not in self.agents:
            print(f"Unknown agent: {agent_name}")
            print("Available agents:", ", ".join(self.agents.keys()))
            return
            
        self.current_agent = agent_name
        print(f"Switched to {agent_name}")
    
    def do_list(self, arg: str) -> None:
        """List available agents."""
        print("\nAvailable agents:")
        for name, agent in self.agents.items():
            print(f"- {name} ({agent.role})")
    
    def do_clear(self, arg: str) -> None:
        """Clear conversation history."""
        agent = self.agents[self.current_agent]
        if agent.memory:
            agent.memory.clear()
        print("Conversation history cleared.")
    
    def do_quit(self, arg: str) -> bool:
        """Exit the program."""
        print("\nGoodbye!")
        return True
    
    def do_EOF(self, arg: str) -> bool:
        """Handle EOF (Ctrl+D)."""
        print("\nGoodbye!")
        return True
    
    def cmdloop(self, intro=None):
        """Override cmdloop to handle keyboard interrupts."""
        try:
            super().cmdloop(intro)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
            return self.do_quit("")
    
    def get_names(self) -> List[str]:
        """Get list of command names."""
        return [n[3:] for n in dir(self) if n.startswith('do_')]
    
    def default(self, line: str) -> None:
        """Handle unknown commands."""
        if self.auto_mode and line:
            self.do_talk(line)
        else:
            print(f"Unknown command: {line}")
            print("Available commands:", ", ".join(self.get_names()))

def main():
    """Run the CLI."""
    cli = AgentCLI()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(cli.initialize())
    cli.cmdloop()

if __name__ == "__main__":
    main()
