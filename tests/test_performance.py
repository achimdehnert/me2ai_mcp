"""Performance tests for the AI coaching system."""
import os
import pytest
import time
import asyncio
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from me2ai.cli import AgentCLI
from me2ai.agents.expert_agents import create_expert_agent
from me2ai.llms.openai_provider import OpenAIProvider
from me2ai.llms.groq_provider import GroqProvider
from me2ai.memory import ConversationMemory

def measure_response_time(func) -> Tuple[float, str]:
    """Measure the response time of a function."""
    start_time = time.time()
    result = func()
    end_time = time.time()
    return end_time - start_time, result

@pytest.mark.performance
def test_agent_response_times():
    """Test response times for different agents."""
    cli = AgentCLI()
    test_message = "Tell me about your expertise"
    response_times = {}
    
    for agent_name, agent in cli.agents.items():
        times = []
        # Test each agent 3 times
        for _ in range(3):
            duration, response = measure_response_time(
                lambda: agent.respond(test_message)
            )
            times.append(duration)
            assert len(response) > 0  # Verify valid response
        
        # Calculate statistics
        response_times[agent_name] = {
            'min': min(times),
            'max': max(times),
            'avg': statistics.mean(times),
            'median': statistics.median(times)
        }
    
    # Log performance metrics
    for agent_name, metrics in response_times.items():
        print(f"\n{agent_name} response times (seconds):")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
        assert metrics['avg'] < 10.0  # Average response should be under 10 seconds

@pytest.mark.performance
def test_memory_performance():
    """Test memory performance with large conversations."""
    memory = ConversationMemory()
    start_time = time.time()
    
    # Add 100 message pairs
    for i in range(100):
        memory.save_context(
            {"input": f"Test message {i}"},
            {"output": f"Test response {i}"}
        )
    
    save_time = time.time() - start_time
    print(f"\nTime to save 100 messages: {save_time:.2f} seconds")
    assert save_time < 1.0  # Should be fast
    
    # Test load time
    start_time = time.time()
    vars = memory.load_memory_variables({})
    load_time = time.time() - start_time
    print(f"Time to load memory variables: {load_time:.2f} seconds")
    assert load_time < 0.1  # Should be very fast

@pytest.mark.performance
def test_cli_command_performance():
    """Test CLI command execution performance."""
    cli = AgentCLI()
    commands = {
        'talk': lambda: cli.do_talk("Hello"),
        'switch': lambda: cli.do_switch("german_professor"),
        'list': lambda: cli.do_list(""),
        'clear': lambda: cli.do_clear(""),
        'help': lambda: cli.do_help("")
    }
    
    command_times = {}
    for cmd_name, cmd_func in commands.items():
        duration, _ = measure_response_time(cmd_func)
        command_times[cmd_name] = duration
        print(f"\n{cmd_name} command execution time: {duration:.4f} seconds")
        assert duration < 1.0  # Commands should be responsive

@pytest.mark.performance
@pytest.mark.slow
def test_concurrent_requests():
    """Test system performance under concurrent requests."""
    cli = AgentCLI()
    agent = cli.agents['seo_expert']
    num_requests = 5
    
    async def make_request(message: str) -> Tuple[float, str]:
        """Make an async request to the agent."""
        loop = asyncio.get_event_loop()
        start_time = time.time()
        response = await loop.run_in_executor(
            None, agent.respond, message
        )
        end_time = time.time()
        return end_time - start_time, response
    
    async def run_concurrent_requests():
        """Run multiple requests concurrently."""
        tasks = []
        for i in range(num_requests):
            task = make_request(f"Question {i} about SEO")
            tasks.append(task)
        return await asyncio.gather(*tasks)
    
    # Run concurrent requests
    results = asyncio.run(run_concurrent_requests())
    
    # Analyze results
    times = [r[0] for r in results]
    responses = [r[1] for r in results]
    
    print(f"\nConcurrent request times (seconds):")
    print(f"  Min: {min(times):.2f}")
    print(f"  Max: {max(times):.2f}")
    print(f"  Average: {statistics.mean(times):.2f}")
    
    # Verify all responses are valid
    assert all(len(r) > 0 for r in responses)
    # Verify reasonable response times
    assert statistics.mean(times) < 15.0  # Average should be under 15 seconds
