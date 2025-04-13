"""Load testing for the agent system."""
import asyncio
import pytest
import time
from typing import Dict, Any, List
from cli import AgentCLI
from memory import ConversationMemory

@pytest.fixture
async def cli():
    """Create a CLI instance for testing."""
    cli = AgentCLI()
    await cli.initialize()
    return cli

async def send_request(
    cli: AgentCLI,
    agent_name: str,
    message: str,
    delay: float = 0.0
) -> Dict[str, Any]:
    """Send a request to an agent and measure performance."""
    start_time = time.time()
    
    try:
        # Switch to agent if needed
        if cli.current_agent != agent_name:
            cli.do_switch(agent_name)
        
        # Send message
        agent = cli.agents[agent_name]
        response = await agent.respond(message)
        
        # Add delay if specified
        if delay > 0:
            await asyncio.sleep(delay)
            
        end_time = time.time()
        return {
            "success": True,
            "response_time": end_time - start_time,
            "agent": agent_name,
            "message": message,
            "response": response
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "agent": agent_name,
            "message": message
        }

async def run_concurrent_requests(
    cli: AgentCLI,
    agent_name: str,
    num_requests: int,
    concurrent_requests: int,
    delay: float = 0.0
) -> List[Dict[str, Any]]:
    """Run multiple requests concurrently."""
    tasks = []
    results = []
    
    # Create message templates
    messages = [
        f"Test message {i} for load testing" for i in range(num_requests)
    ]
    
    # Process requests in batches
    for i in range(0, num_requests, concurrent_requests):
        batch = messages[i:i + concurrent_requests]
        batch_tasks = [
            send_request(cli, agent_name, msg, delay) for msg in batch
        ]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        
        # Add delay between batches if specified
        if delay > 0:
            await asyncio.sleep(delay)
    
    return results

@pytest.mark.load
@pytest.mark.slow
async def test_moderate_load(cli):
    """Test system under moderate load."""
    metrics = await run_concurrent_requests(
        cli,
        agent_name="german_professor",
        num_requests=10,
        concurrent_requests=2,
        delay=0.5
    )
    
    assert len(metrics) == 10
    assert all(r["success"] for r in metrics)

@pytest.mark.load
@pytest.mark.slow
async def test_heavy_load(cli):
    """Test system under heavy load."""
    metrics = await run_concurrent_requests(
        cli,
        agent_name="german_professor",
        num_requests=20,
        concurrent_requests=5,
        delay=0.2
    )
    
    assert len(metrics) == 20
    assert all(r["success"] for r in metrics)

@pytest.mark.load
@pytest.mark.slow
async def test_mixed_agent_load(cli):
    """Test multiple agents under load."""
    agents = ["german_professor", "dating_expert", "seo_expert"]
    all_metrics = []
    
    for agent in agents:
        metrics = await run_concurrent_requests(
            cli,
            agent_name=agent,
            num_requests=5,
            concurrent_requests=2,
            delay=0.5
        )
        all_metrics.extend(metrics)
    
    assert len(all_metrics) == len(agents) * 5
    assert all(r["success"] for r in all_metrics)

@pytest.mark.load
@pytest.mark.slow
async def test_memory_load(cli):
    """Test memory system under load."""
    # Send multiple messages to build up memory
    messages = [
        "Hello, how are you?",
        "Can you help me learn German?",
        "What's the best way to practice?",
        "How do I improve my pronunciation?",
        "Thank you for your help!"
    ]
    
    for msg in messages:
        await send_request(cli, "german_professor", msg)
    
    # Verify memory state
    agent = cli.agents["german_professor"]
    assert isinstance(agent.memory, ConversationMemory)
    assert len(agent.memory.messages) > 0
