"""Tests for SEO workflow functionality."""
import pytest
from datetime import datetime
from typing import Dict, Any
from me2ai.tools.seo.workflow.engine import (
    WorkflowStep,
    WorkflowTemplate,
    CustomizedWorkflow,
    WorkflowEngine
)

def dummy_action(context: Dict[str, Any]) -> Dict[str, Any]:
    """Dummy action for testing."""
    return {"result": "success"}

@pytest.fixture
def sample_step():
    """Sample workflow step for testing."""
    return WorkflowStep(
        name="test_step",
        description="Test step",
        action=dummy_action,
        dependencies=[],
        estimated_time=30,
        user_input_required=False,
        customizable=True
    )

@pytest.fixture
def sample_template():
    """Sample workflow template for testing."""
    return WorkflowTemplate(
        name="Test Workflow",
        description="Test workflow template",
        steps=["step1", "step2"],
        estimated_duration=60,
        required_tools=["tool1", "tool2"],
        customization_points=["point1", "point2"]
    )

class TestWorkflowStep:
    """Test workflow step functionality."""
    
    def test_step_creation(self, sample_step):
        """Test creating workflow step."""
        assert sample_step.name == "test_step"
        assert sample_step.estimated_time == 30
        assert not sample_step.user_input_required
        assert sample_step.customizable
    
    def test_step_execution(self, sample_step):
        """Test executing workflow step."""
        result = sample_step.execute({})
        assert result["result"] == "success"
    
    def test_dependencies(self):
        """Test step dependencies."""
        step = WorkflowStep(
            name="dependent_step",
            description="Step with dependencies",
            action=dummy_action,
            dependencies=["step1", "step2"]
        )
        assert len(step.dependencies) == 2
        assert "step1" in step.dependencies

class TestWorkflowTemplate:
    """Test workflow template functionality."""
    
    def test_template_creation(self, sample_template):
        """Test creating workflow template."""
        assert sample_template.name == "Test Workflow"
        assert len(sample_template.steps) == 2
        assert len(sample_template.required_tools) == 2
    
    def test_estimated_duration(self, sample_template):
        """Test template duration calculation."""
        assert sample_template.estimated_duration == 60
    
    def test_customization_points(self, sample_template):
        """Test template customization points."""
        assert len(sample_template.customization_points) == 2
        assert "point1" in sample_template.customization_points

class TestCustomizedWorkflow:
    """Test customized workflow functionality."""
    
    def test_workflow_customization(self):
        """Test customizing workflow."""
        workflow = CustomizedWorkflow(
            base_template="technical_audit",
            added_steps=["new_step"],
            removed_steps=["old_step"],
            modified_steps={
                "step1": {"estimated_time": 45}
            },
            user_preferences={"preference1": "value1"}
        )
        assert workflow.base_template == "technical_audit"
        assert len(workflow.added_steps) == 1
        assert len(workflow.removed_steps) == 1
        assert len(workflow.modified_steps) == 1
    
    def test_invalid_modifications(self):
        """Test invalid workflow modifications."""
        with pytest.raises(ValueError):
            CustomizedWorkflow(
                base_template="invalid_template"
            )

class TestWorkflowEngine:
    """Test workflow engine functionality."""
    
    def setup_method(self):
        """Set up workflow engine."""
        self.engine = WorkflowEngine()
    
    def test_load_templates(self):
        """Test loading default templates."""
        templates = self.engine.templates
        assert "technical_audit" in templates
        assert "content_audit" in templates
    
    def test_create_custom_workflow(self):
        """Test creating custom workflow."""
        workflow = self.engine.create_custom_workflow(
            base_template="technical_audit",
            customizations={
                "added_steps": ["new_step"],
                "user_preferences": {"pref1": "val1"}
            }
        )
        assert workflow.base_template == "technical_audit"
        assert len(workflow.added_steps) == 1
    
    def test_execute_workflow(self):
        """Test executing workflow."""
        workflow = CustomizedWorkflow(
            base_template="technical_audit"
        )
        results = self.engine.execute_workflow(
            workflow,
            context={"start_url": "https://example.com"}
        )
        assert isinstance(results, dict)
    
    def test_step_ordering(self):
        """Test workflow step ordering."""
        steps = [
            WorkflowStep(
                name="step2",
                description="Step 2",
                action=dummy_action,
                dependencies=["step1"]
            ),
            WorkflowStep(
                name="step1",
                description="Step 1",
                action=dummy_action
            )
        ]
        ordered = self.engine._order_steps(steps)
        assert ordered[0].name == "step1"
        assert ordered[1].name == "step2"
    
    def test_user_input_handling(self):
        """Test handling user input in workflow."""
        step = WorkflowStep(
            name="input_step",
            description="Step requiring input",
            action=dummy_action,
            user_input_required=True
        )
        context = {}
        input_data = self.engine._get_user_input(step, context)
        assert isinstance(input_data, dict)

def test_workflow_integration():
    """Test complete workflow integration."""
    engine = WorkflowEngine()
    
    # Create custom workflow
    workflow = engine.create_custom_workflow(
        base_template="technical_audit",
        customizations={
            "added_steps": ["custom_analysis"],
            "modified_steps": {
                "crawl_site": {
                    "estimated_time": 45,
                    "user_input_required": True
                }
            }
        }
    )
    
    # Execute workflow
    results = engine.execute_workflow(
        workflow,
        context={
            "start_url": "https://example.com",
            "crawl_depth": 3
        }
    )
    
    assert isinstance(results, dict)
    assert len(results) > 0
