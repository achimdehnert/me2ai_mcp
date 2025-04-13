"""Performance tests for ME2AI MCP server and tools."""

import pytest
import time
from typing import Dict, Any, List
import statistics

from me2ai_mcp.base import ME2AIMCPServer


class TestServerPerformance:
    """Test the performance characteristics of the ME2AI MCP server."""

    @pytest.fixture
    def server_with_tools(self):
        """Create a server with multiple tools for performance testing."""
        server = ME2AIMCPServer()
        
        # Simple tool with minimal processing
        @server.register_tool
        def simple_tool(value: str) -> Dict[str, Any]:
            """Simple tool for baseline performance."""
            return {"result": f"Processed: {value}"}
        
        # Tool with more complex processing
        @server.register_tool
        def complex_tool(items: List[Dict[str, Any]]) -> Dict[str, Any]:
            """More complex tool for performance comparison."""
            results = []
            for item in items:
                # Simulate some processing
                processed = {k: f"{v}_processed" for k, v in item.items()}
                results.append(processed)
            return {"results": results}
        
        return server
    
    def test_should_meet_performance_requirements_for_simple_tools(self, server_with_tools):
        """Test performance of simple tool operations."""
        # Performance threshold in seconds
        threshold = 0.005
        
        # Run multiple iterations to get stable measurements
        iterations = 100
        execution_times = []
        
        for i in range(iterations):
            start_time = time.time()
            server_with_tools.execute_tool("simple_tool", {"value": f"test_{i}"})
            end_time = time.time()
            execution_times.append(end_time - start_time)
        
        avg_time = statistics.mean(execution_times)
        max_time = max(execution_times)
        
        # Assert performance requirements
        assert avg_time < threshold, f"Average execution time {avg_time:.6f}s exceeds threshold {threshold:.6f}s"
        
        # Output performance statistics
        print(f"Simple tool performance: Avg={avg_time:.6f}s, Max={max_time:.6f}s, Min={min(execution_times):.6f}s")
    
    def test_should_scale_linearly_with_input_size(self, server_with_tools):
        """Test how tool performance scales with input size."""
        # Define input sizes to test
        input_sizes = [10, 100, 1000]
        
        results = {}
        for size in input_sizes:
            # Create input data of specified size
            items = [{"key": f"value_{i}"} for i in range(size)]
            
            # Measure execution time
            start_time = time.time()
            server_with_tools.execute_tool("complex_tool", {"items": items})
            end_time = time.time()
            
            execution_time = end_time - start_time
            results[size] = execution_time
            
            # Output result
            print(f"Input size {size}: {execution_time:.6f}s")
        
        # Check for linear scaling (time increase should be roughly proportional to size increase)
        # This is a simplified check - in reality you might want more sophisticated analysis
        smallest_size = min(input_sizes)
        smallest_time = results[smallest_size]
        
        for size in input_sizes[1:]:
            expected_ratio = size / smallest_size
            actual_ratio = results[size] / smallest_time
            
            # Allow for some variance (50% tolerance)
            assert actual_ratio < expected_ratio * 1.5, \
                f"Performance scaling is worse than expected at size {size}"
    
    def test_should_handle_parallel_tool_execution_efficiently(self, server_with_tools):
        """Test performance with parallel tool execution."""
        # This test simulates parallel execution by running tools in quick succession
        
        # Performance threshold for total execution
        threshold = 0.1
        
        # Run multiple tools in sequence
        start_time = time.time()
        
        # Execute simple tool multiple times
        for i in range(10):
            server_with_tools.execute_tool("simple_tool", {"value": f"parallel_{i}"})
        
        # Execute complex tool
        items = [{"key": f"value_{i}"} for i in range(20)]
        server_with_tools.execute_tool("complex_tool", {"items": items})
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Assert performance
        assert total_time < threshold, \
            f"Total execution time {total_time:.6f}s exceeds threshold {threshold:.6f}s"
        
        print(f"Parallel execution time: {total_time:.6f}s")
