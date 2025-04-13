"""
Performance tests for the ME2AI MCP framework.
Tests benchmark execution times and memory usage for various operations.
"""
import pytest
import asyncio
import time
import os
from unittest.mock import patch, Mock, MagicMock
import tempfile
import random
import string
import gc
import sys
import json
from pathlib import Path

from me2ai_mcp.base import ME2AIMCPServer, BaseTool, register_tool
from me2ai_mcp.tools.web import WebFetchTool
from me2ai_mcp.tools.github import GitHubRepositoryTool


# Helper function to generate large text
def generate_large_text(size_kb):
    """Generate random text of specified size in KB."""
    chars = string.ascii_letters + string.digits + string.punctuation + ' ' * 10
    # Convert KB to characters (approximation)
    char_count = size_kb * 1024
    return ''.join(random.choice(chars) for _ in range(char_count))


# Helper function to measure memory usage
def get_memory_usage():
    """Return current memory usage in MB."""
    gc.collect()
    if sys.platform == 'win32':
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    else:
        # Fallback for non-Windows platforms
        return 0  # Return 0 when we can't measure


class TestPerformance:
    """Performance tests for ME2AI MCP components."""

    @pytest.mark.asyncio
    async def test_web_fetch_large_content_performance(self):
        """Test web fetch performance with large content."""
        # Skip if on CI environment
        if "CI" in os.environ:
            pytest.skip("Skipping performance test in CI environment")
            
        tool = WebFetchTool()
        
        # Create large HTML content sizes
        sizes = [100, 500, 1000]  # in KB
        times = {}
        memory_before = get_memory_usage()
        
        for size in sizes:
            # Generate large HTML
            large_html = f"<html><body>{'<p>' + generate_large_text(size) + '</p>'}</body></html>"
            
            # Mock response with large content
            mock_response = Mock()
            mock_response.text = large_html
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.raise_for_status = Mock()
            
            with patch('requests.get', return_value=mock_response):
                # Measure execution time
                start_time = time.time()
                result = await tool.execute({"url": "https://example.com"})
                end_time = time.time()
                
                # Record time
                execution_time = end_time - start_time
                times[size] = execution_time
                
                # Verify success
                assert result["success"] is True
                assert result["content_type"] == "text/html"
                assert len(result["content"]) >= size * 1024  # Approximate size check
        
        memory_after = get_memory_usage()
        memory_diff = memory_after - memory_before
        
        # Print performance results
        print(f"\nWeb fetch performance:")
        for size, execution_time in times.items():
            print(f"  {size} KB: {execution_time:.4f} seconds")
        print(f"  Memory usage difference: {memory_diff:.2f} MB")
        
        # Basic performance assertions
        # Expect larger sizes to take longer, but not excessively
        if len(sizes) > 1:
            assert times[sizes[0]] <= times[sizes[-1]]  # First size should be fastest
            # Check execution time ratio is reasonable (should not grow faster than linear)
            ratio = times[sizes[-1]] / times[sizes[0]]
            size_ratio = sizes[-1] / sizes[0]
            assert ratio <= size_ratio * 2  # Allow for some overhead, but not excessive

    @pytest.mark.asyncio
    async def test_parallel_tool_execution_performance(self):
        """Test performance of parallel tool execution."""
        # Skip if on CI environment
        if "CI" in os.environ:
            pytest.skip("Skipping performance test in CI environment")
            
        # Create a server
        server = ME2AIMCPServer(server_name="benchmark-server")
        
        # Define a test tool with controllable execution time
        @register_tool(server)
        class DelayTool(BaseTool):
            """Tool with configurable delay."""
            
            name = "delay_tool"
            description = "A tool that waits for a specified time"
            
            async def execute(self, parameters):
                # Get delay in seconds
                delay = parameters.get("delay", 0.1)
                # Simulate work by sleeping
                await asyncio.sleep(delay)
                return {"success": True, "delay": delay}
        
        # Get tool instance
        delay_tool = server.tools["delay_tool"]
        
        # Test sequential execution
        sequential_start = time.time()
        sequential_results = []
        
        for i in range(5):
            result = await delay_tool.execute({"delay": 0.1})
            sequential_results.append(result)
            
        sequential_time = time.time() - sequential_start
        
        # Test parallel execution
        parallel_start = time.time()
        
        tasks = [
            delay_tool.execute({"delay": 0.1})
            for _ in range(5)
        ]
        
        parallel_results = await asyncio.gather(*tasks)
        
        parallel_time = time.time() - parallel_start
        
        # Print performance results
        print(f"\nTool execution performance:")
        print(f"  Sequential execution time: {sequential_time:.4f} seconds")
        print(f"  Parallel execution time: {parallel_time:.4f} seconds")
        print(f"  Speedup: {sequential_time / parallel_time:.2f}x")
        
        # Verify all results were successful
        assert all(result["success"] for result in sequential_results)
        assert all(result["success"] for result in parallel_results)
        
        # Parallel should be faster than sequential
        # Allow some overhead for task creation
        assert parallel_time < sequential_time * 0.8

    @pytest.mark.asyncio
    async def test_github_api_performance(self):
        """Test GitHub API performance with mock data."""
        # Skip if on CI environment
        if "CI" in os.environ:
            pytest.skip("Skipping performance test in CI environment")
            
        # Create GitHub repository tool
        tool = GitHubRepositoryTool()
        
        # Generate mock repository data of different sizes
        repo_counts = [10, 50, 100]
        times = {}
        
        for count in repo_counts:
            # Generate mock repositories
            repositories = [
                {
                    "full_name": f"user/repo{i}",
                    "html_url": f"https://github.com/user/repo{i}",
                    "description": f"Test repository {i} with some description text",
                    "stargazers_count": random.randint(0, 1000),
                    "forks_count": random.randint(0, 300),
                    "open_issues_count": random.randint(0, 100),
                    "language": random.choice(["Python", "JavaScript", "Java", "C++", "Go"]),
                    "created_at": "2022-01-01T00:00:00Z",
                    "updated_at": "2022-02-01T00:00:00Z",
                    "topics": [
                        random.choice(["web", "api", "database", "frontend", "backend"]) 
                        for _ in range(random.randint(1, 5))
                    ]
                }
                for i in range(count)
            ]
            
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "items": repositories,
                "total_count": count
            }
            
            with patch('requests.get', return_value=mock_response):
                with patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}):
                    # Measure execution time
                    start_time = time.time()
                    result = await tool.execute({
                        "operation": "search",
                        "query": "test repo"
                    })
                    end_time = time.time()
                    
                    # Record time
                    execution_time = end_time - start_time
                    times[count] = execution_time
                    
                    # Verify success
                    assert result["success"] is True
                    assert len(result["repositories"]) == count
        
        # Print performance results
        print(f"\nGitHub API performance:")
        for count, execution_time in times.items():
            print(f"  {count} repositories: {execution_time:.4f} seconds")
        
        # Check if processing time is reasonable for the data volume
        if len(repo_counts) > 1:
            # Processing time should not grow excessively with repository count
            ratio = times[repo_counts[-1]] / times[repo_counts[0]]
            count_ratio = repo_counts[-1] / repo_counts[0]
            # Allow reasonable overhead but not excessive
            assert ratio <= max(1.5, count_ratio * 0.2)

    def test_file_io_performance(self):
        """Test file I/O performance with different file sizes."""
        # Skip if on CI environment
        if "CI" in os.environ:
            pytest.skip("Skipping performance test in CI environment")
            
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files of different sizes
            sizes = [100, 500, 1000]  # in KB
            files = {}
            
            for size in sizes:
                file_path = temp_path / f"test_file_{size}kb.txt"
                file_content = generate_large_text(size)
                file_path.write_text(file_content)
                files[size] = file_path
            
            # Test read performance
            read_times = {}
            
            for size, file_path in files.items():
                start_time = time.time()
                
                # Read file
                content = file_path.read_text()
                
                end_time = time.time()
                read_times[size] = end_time - start_time
                
                # Verify content length
                assert len(content) >= size * 1024  # Approximate size check
            
            # Test write performance
            write_times = {}
            
            for size in sizes:
                # Generate new content
                new_content = generate_large_text(size)
                
                # Create a new file for writing
                new_file_path = temp_path / f"new_file_{size}kb.txt"
                
                start_time = time.time()
                
                # Write file
                new_file_path.write_text(new_content)
                
                end_time = time.time()
                write_times[size] = end_time - start_time
                
                # Verify file exists
                assert new_file_path.exists()
            
            # Print performance results
            print(f"\nFile I/O performance:")
            print(f"  Read times:")
            for size, time_taken in read_times.items():
                print(f"    {size} KB: {time_taken:.4f} seconds")
            
            print(f"  Write times:")
            for size, time_taken in write_times.items():
                print(f"    {size} KB: {time_taken:.4f} seconds")
            
            # Basic performance assertions
            if len(sizes) > 1:
                # Larger files should take longer to read and write
                assert read_times[sizes[0]] <= read_times[sizes[-1]]
                assert write_times[sizes[0]] <= write_times[sizes[-1]]


class TestServerPerformance:
    """Performance tests for ME2AI MCP server."""

    @pytest.mark.asyncio
    async def test_server_concurrent_requests(self):
        """Test server performance with concurrent requests."""
        # Skip if on CI environment
        if "CI" in os.environ:
            pytest.skip("Skipping performance test in CI environment")
            
        # Create a server
        server = ME2AIMCPServer(server_name="concurrent-test-server")
        
        # Define a test tool with controllable execution time
        @register_tool(server)
        class EchoTool(BaseTool):
            """Echo tool that returns its input."""
            
            name = "echo_tool"
            description = "A tool that returns its input after a delay"
            
            async def execute(self, parameters):
                # Get input and delay
                input_data = parameters.get("input", "")
                delay = parameters.get("delay", 0.05)
                
                # Simulate work by sleeping
                await asyncio.sleep(delay)
                
                return {"success": True, "result": input_data}
        
        # Setup mock requests
        request_counts = [10, 50, 100]
        concurrency_levels = [1, 5, 10]
        
        results = {}
        
        for count in request_counts:
            for concurrency in concurrency_levels:
                # Skip high concurrency for large request counts to avoid test timeout
                if count * concurrency > 500:
                    continue
                
                # Create a list of mock requests
                mock_requests = []
                for i in range(count):
                    mock_request = Mock()
                    mock_request.json = Mock(return_value={
                        "tool": "echo_tool",
                        "parameters": {"input": f"input{i}", "delay": 0.05}
                    })
                    mock_requests.append(mock_request)
                
                # Measure execution time with limited concurrency
                start_time = time.time()
                
                # Set up tasks
                semaphore = asyncio.Semaphore(concurrency)
                
                async def process_request(req):
                    async with semaphore:
                        with patch.object(server, 'send_response'):
                            await server.handle_request(req)
                
                tasks = [process_request(req) for req in mock_requests]
                await asyncio.gather(*tasks)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Record result
                key = f"{count}_{concurrency}"
                results[key] = {
                    "count": count,
                    "concurrency": concurrency,
                    "time": execution_time,
                    "requests_per_second": count / execution_time
                }
        
        # Print performance results
        print(f"\nServer concurrent request performance:")
        for key, result in results.items():
            print(f"  {result['count']} requests, concurrency {result['concurrency']}:")
            print(f"    Time: {result['time']:.4f} seconds")
            print(f"    Requests/second: {result['requests_per_second']:.2f}")
        
        # Verify performance scaling with concurrency
        for count in request_counts:
            # Skip if not all concurrency levels were tested for this count
            available_keys = [key for key in results if key.startswith(f"{count}_")]
            if len(available_keys) < 2:
                continue
                
            # Get results for this count with different concurrency levels
            concurrency_results = {int(key.split('_')[1]): results[key]["time"] for key in available_keys}
            
            # Higher concurrency should generally be faster
            concurrency_values = sorted(concurrency_results.keys())
            for i in range(len(concurrency_values) - 1):
                c1 = concurrency_values[i]
                c2 = concurrency_values[i + 1]
                
                # Only compare if there's a significant concurrency difference
                if c2 >= c1 * 2:
                    # Higher concurrency should generally be faster, allowing for some variability
                    assert concurrency_results[c1] >= concurrency_results[c2] * 0.7, \
                        f"Expected concurrency {c2} to be faster than {c1} for {count} requests"


class TestRequestRateThrottling:
    """Tests for API request rate throttling."""

    @pytest.mark.asyncio
    async def test_api_rate_limiting(self):
        """Test API rate limiting for external services."""
        # Skip if on CI environment
        if "CI" in os.environ:
            pytest.skip("Skipping performance test in CI environment")
            
        # Create a server
        server = ME2AIMCPServer(server_name="rate-limit-test-server")
        
        # Define a tool with rate limiting
        @register_tool(server)
        class RateLimitedTool(BaseTool):
            """Tool with rate limiting."""
            
            name = "rate_limited_tool"
            description = "A tool with rate limiting"
            
            def __init__(self, requests_per_minute=60):
                super().__init__()
                self.requests_per_minute = requests_per_minute
                self.min_interval = 60 / requests_per_minute  # seconds per request
                self.last_request_time = 0
            
            async def execute(self, parameters):
                # Apply rate limiting
                current_time = time.time()
                elapsed = current_time - self.last_request_time
                
                if elapsed < self.min_interval and self.last_request_time > 0:
                    # Wait until we can make a request
                    await asyncio.sleep(max(0, self.min_interval - elapsed))
                
                # Update last request time
                self.last_request_time = time.time()
                
                # Simulate API call
                return {"success": True, "timestamp": time.time()}
        
        # Get tool instance with 30 requests per minute (1 request per 2 seconds)
        tool = RateLimitedTool(requests_per_minute=30)
        
        # Make multiple requests in quick succession
        num_requests = 5
        start_time = time.time()
        
        results = []
        for i in range(num_requests):
            result = await tool.execute({})
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify success of all requests
        assert all(result["success"] for result in results)
        
        # Calculate expected time
        expected_min_time = (num_requests - 1) * (60 / 30)  # (n-1) * interval
        
        # Print performance results
        print(f"\nAPI rate limiting test:")
        print(f"  Requests per minute: 30")
        print(f"  Number of requests: {num_requests}")
        print(f"  Total time: {total_time:.4f} seconds")
        print(f"  Expected minimum time: {expected_min_time:.4f} seconds")
        
        # Verify total time is at least the expected minimum
        assert total_time >= expected_min_time * 0.9  # Allow for small timing variations
        
        # Verify timestamps between requests have minimum interval
        for i in range(1, len(results)):
            interval = results[i]["timestamp"] - results[i-1]["timestamp"]
            assert interval >= (60 / 30) * 0.9  # Allow for small timing variations
