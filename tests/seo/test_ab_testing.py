"""Tests for SEO A/B testing functionality."""
import pytest
from datetime import datetime, timedelta
import numpy as np
from me2ai.tools.seo.testing.ab_testing import (
    TestVariant,
    TestResult,
    ABTest,
    ABTestingTool,
    SEOTestGenerator
)

@pytest.fixture
def sample_variants():
    """Sample test variants for testing."""
    return [
        TestVariant(
            name="control",
            changes={"title": "Original Title"},
            metrics={"conversion_rate": 0.1},
            sample_size=1000
        ),
        TestVariant(
            name="treatment",
            changes={"title": "New Title"},
            metrics={"conversion_rate": 0.12},
            sample_size=1000
        )
    ]

@pytest.fixture
def sample_test(sample_variants):
    """Sample A/B test for testing."""
    return ABTest(
        name="Test Experiment",
        hypothesis="New title will improve CTR",
        variants=sample_variants,
        duration=timedelta(days=14),
        start_date=datetime.now()
    )

class TestABTestingTool:
    """Test A/B testing tool functionality."""
    
    def setup_method(self):
        """Set up testing tool."""
        self.tool = ABTestingTool()
    
    def test_create_test(self):
        """Test creating new A/B test."""
        test = self.tool.create_test(
            name="Title Test",
            hypothesis="New title improves CTR",
            variants=[
                {
                    "name": "control",
                    "changes": {"title": "Original"},
                    "metrics": {},
                    "sample_size": 0
                },
                {
                    "name": "treatment",
                    "changes": {"title": "New"},
                    "metrics": {},
                    "sample_size": 0
                }
            ],
            duration_days=14
        )
        assert test.name == "Title Test"
        assert len(test.variants) == 2
        assert test.duration == timedelta(days=14)
    
    def test_analyze_results(self, sample_test):
        """Test analyzing test results."""
        results = self.tool.analyze_results(sample_test)
        assert isinstance(results, TestResult)
        assert results.confidence > 0
        assert results.improvement != 0
        assert len(results.metrics_comparison) == 2
    
    def test_statistical_significance(self):
        """Test statistical significance calculation."""
        variants = [
            TestVariant(
                name="control",
                changes={},
                metrics={"conversion_rate": 0.1},
                sample_size=10000
            ),
            TestVariant(
                name="treatment",
                changes={},
                metrics={"conversion_rate": 0.15},  # 50% improvement
                sample_size=10000
            )
        ]
        test = ABTest(
            name="Significant Test",
            hypothesis="Treatment is better",
            variants=variants,
            duration=timedelta(days=14),
            start_date=datetime.now()
        )
        results = self.tool.analyze_results(test)
        assert results.winner == "treatment"
        assert results.confidence > 95

class TestSEOTestGenerator:
    """Test SEO test generator functionality."""
    
    def setup_method(self):
        """Set up test generator."""
        self.generator = SEOTestGenerator()
    
    def test_generate_title_test(self):
        """Test generating title test."""
        current_title = "Original Title"
        test = self.generator.generate_title_test(current_title)
        assert test.name == "Title Tag Optimization"
        assert len(test.variants) == 2
        assert test.variants[0].changes["title"] == current_title
        assert test.variants[1].changes["title"] != current_title
    
    def test_generate_meta_description_test(self):
        """Test generating meta description test."""
        current_description = "Original description"
        test = self.generator.generate_meta_description_test(current_description)
        assert test.name == "Meta Description Optimization"
        assert len(test.variants) == 2
        assert test.variants[0].changes["description"] == current_description
        assert test.variants[1].changes["description"] != current_description

class TestTestVariant:
    """Test test variant functionality."""
    
    def test_variant_creation(self):
        """Test creating test variant."""
        variant = TestVariant(
            name="test",
            changes={"field": "value"},
            metrics={"metric": 1.0},
            sample_size=100
        )
        assert variant.name == "test"
        assert variant.changes["field"] == "value"
        assert variant.metrics["metric"] == 1.0
        assert variant.sample_size == 100
    
    def test_invalid_metrics(self):
        """Test handling invalid metrics."""
        with pytest.raises(ValueError):
            TestVariant(
                name="test",
                changes={},
                metrics={"invalid": "not a number"},
                sample_size=100
            )

class TestTestResult:
    """Test test result functionality."""
    
    def test_result_creation(self):
        """Test creating test result."""
        result = TestResult(
            winner="treatment",
            confidence=95.0,
            improvement=10.0,
            metrics_comparison={
                "control": {"rate": 0.1},
                "treatment": {"rate": 0.11}
            },
            recommendation="Implement treatment"
        )
        assert result.winner == "treatment"
        assert result.confidence == 95.0
        assert result.improvement == 10.0
    
    def test_no_winner(self):
        """Test result with no clear winner."""
        result = TestResult(
            winner=None,
            confidence=80.0,  # Below threshold
            improvement=5.0,
            metrics_comparison={},
            recommendation="Continue testing"
        )
        assert result.winner is None
        assert "Continue testing" in result.recommendation
