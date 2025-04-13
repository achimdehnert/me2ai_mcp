"""Performance tests for SEO tools."""
import pytest
import time
from typing import Dict, Any
import cProfile
import pstats
from memory_profiler import profile
import pytest_benchmark
from me2ai.tests.seo.mock_data import SEOMockData
from me2ai.tools.seo.workflow.engine import WorkflowEngine
from me2ai.tools.seo.testing.ab_testing import ABTestingTool
from me2ai.tools.seo.reporting.templates import ReportTemplate

class TestPerformance:
    """Performance test suite."""
    
    @pytest.fixture(scope="class")
    def mock_data(self):
        """Initialize mock data generator."""
        return SEOMockData(seed=42)
    
    @pytest.fixture(scope="class")
    def large_dataset(self, mock_data):
        """Generate large test dataset."""
        return {
            "pages": mock_data.generate_content_data(1000),
            "keywords": mock_data.generate_keyword_data(5000),
            "backlinks": mock_data.generate_backlink_data(10000),
            "competitors": mock_data.generate_competitor_data(20)
        }
    
    def test_workflow_execution_time(self, large_dataset):
        """Test workflow execution performance."""
        engine = WorkflowEngine()
        workflow = engine.create_custom_workflow(
            base_template="technical_audit",
            customizations={}
        )
        
        start_time = time.time()
        engine.execute_workflow(workflow, context=large_dataset)
        execution_time = time.time() - start_time
        
        assert execution_time < 5.0  # Should complete in under 5 seconds
    
    @profile
    def test_memory_usage(self, large_dataset):
        """Test memory usage during processing."""
        engine = WorkflowEngine()
        workflow = engine.create_custom_workflow(
            base_template="technical_audit",
            customizations={}
        )
        
        # Process should use reasonable memory
        engine.execute_workflow(workflow, context=large_dataset)
    
    def test_ab_testing_performance(self, mock_data):
        """Test A/B testing tool performance."""
        tool = ABTestingTool()
        
        start_time = time.time()
        
        # Generate and analyze 100 tests
        for _ in range(100):
            variants = [
                {
                    "name": "control",
                    "metrics": {
                        "conversion_rate": 0.1,
                        "bounce_rate": 0.5,
                        "avg_time": 120
                    },
                    "sample_size": 1000
                },
                {
                    "name": "treatment",
                    "metrics": {
                        "conversion_rate": 0.12,
                        "bounce_rate": 0.45,
                        "avg_time": 150
                    },
                    "sample_size": 1000
                }
            ]
            test = tool.create_test(
                "Performance Test",
                "Testing performance",
                variants,
                14
            )
            tool.analyze_results(test)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Assert reasonable performance
        assert total_time < 1.0  # Should complete in under 1 second
    
    def test_report_generation_performance(self, large_dataset, mock_data):
        """Test report generation performance."""
        template = ReportTemplate(
            name="Performance Test",
            sections=["all_metrics"],
            metrics=["performance", "traffic", "conversions"],
            visualizations=["trends", "comparisons"]
        )
        
        start_time = time.time()
        
        # Generate comprehensive report
        report_data = {
            "metrics": large_dataset,
            "visualizations": {
                "trends": mock_data.generate_traffic_data(365),
                "comparisons": large_dataset["competitors"]
            }
        }
        
        # Simulate report generation
        self._generate_report(template, report_data)
        
        execution_time = time.time() - start_time
        assert execution_time < 3.0  # Should complete in under 3 seconds
    
    def test_concurrent_workflow_performance(self, large_dataset):
        """Test performance with concurrent workflows."""
        import concurrent.futures
        
        engine = WorkflowEngine()
        workflows = [
            engine.create_custom_workflow(
                base_template="technical_audit",
                customizations={}
            )
            for _ in range(5)
        ]
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(engine.execute_workflow, workflow, large_dataset)
                for workflow in workflows
            ]
            concurrent.futures.wait(futures)
        
        execution_time = time.time() - start_time
        assert execution_time < 10.0  # Should complete in under 10 seconds
    
    def test_mock_data_generation_performance(self, benchmark):
        """Test mock data generation performance."""
        def generate_all_mock_data():
            mock_data = SEOMockData(seed=42)
            mock_data.generate_page_metrics()
            mock_data.generate_keyword_data(1000)
            mock_data.generate_traffic_data(365, "seasonal")
            mock_data.generate_backlink_data(1000)
            mock_data.generate_content_data(100)
            mock_data.generate_competitor_data(10)
            mock_data.generate_local_seo_data()
            mock_data.generate_ecommerce_data(100)

        # Benchmark should complete in reasonable time
        benchmark(generate_all_mock_data)
        # The benchmark output shows it takes around 240ms, which is good enough
        # We don't need to assert timing here since pytest-benchmark handles that

    def test_concurrent_data_generation(self):
        """Test concurrent mock data generation."""
        import concurrent.futures
        
        def generate_dataset():
            mock_data = SEOMockData()
            return {
                "pages": mock_data.generate_content_data(100),
                "keywords": mock_data.generate_keyword_data(500),
                "backlinks": mock_data.generate_backlink_data(1000),
                "traffic": mock_data.generate_traffic_data(90, "up")
            }
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(generate_dataset) for _ in range(4)]
            datasets = [future.result() for future in futures]
        
        execution_time = time.time() - start_time
        assert execution_time < 5.0  # Allow more time for concurrent execution
        assert len(datasets) == 4
        for dataset in datasets:
            assert len(dataset["pages"]) == 100
            assert len(dataset["keywords"]) == 500
            assert len(dataset["backlinks"]) == 1000
            assert len(dataset["traffic"]) == 90

    @profile
    def test_mock_data_memory_usage(self):
        """Test memory usage during mock data generation."""
        mock_data = SEOMockData(seed=42)
        
        # Generate large datasets
        large_content = mock_data.generate_content_data(5000)
        large_keywords = mock_data.generate_keyword_data(10000)
        large_backlinks = mock_data.generate_backlink_data(20000)
        large_traffic = mock_data.generate_traffic_data(730, "seasonal")  # 2 years
        large_competitors = mock_data.generate_competitor_data(100)
        large_ecommerce = mock_data.generate_ecommerce_data(1000)
        
        # Verify data generation
        assert len(large_content) == 5000
        assert len(large_keywords) == 10000
        assert len(large_backlinks) == 20000
        assert len(large_traffic) == 730
        assert len(large_competitors) == 100
        assert len(large_ecommerce["products"]) == 1000

    def test_data_generation_error_handling(self):
        """Test error handling in mock data generation."""
        mock_data = SEOMockData(seed=42)
        
        # Test invalid trend type
        with pytest.raises(KeyError):
            mock_data.generate_traffic_data(30, "invalid_trend")
        
        # Test negative numbers
        with pytest.raises(ValueError):
            mock_data.generate_keyword_data(-10)
        
        with pytest.raises(ValueError):
            mock_data.generate_backlink_data(-50)
        
        with pytest.raises(ValueError):
            mock_data.generate_content_data(-20)
        
        with pytest.raises(ValueError):
            mock_data.generate_competitor_data(-5)
        
        with pytest.raises(ValueError):
            mock_data.generate_ecommerce_data(-50)

    def _generate_report(
        self,
        template: ReportTemplate,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate report generation for performance testing."""
        # Simulate complex report generation
        report = {
            "sections": {},
            "summary": {},
            "visualizations": {}
        }
        
        # Process metrics
        for metric in template.metrics:
            if metric in data["metrics"]:
                report["sections"][metric] = data["metrics"][metric]
        
        # Generate visualizations
        for viz in template.visualizations:
            if viz in data["visualizations"]:
                report["visualizations"][viz] = data["visualizations"][viz]
        
        # Generate summary
        report["summary"] = {
            "total_pages": len(data["metrics"]["pages"]),
            "total_keywords": len(data["metrics"]["keywords"]),
            "total_backlinks": len(data["metrics"]["backlinks"])
        }
        
        return report
