"""Flexible SEO workflow engine."""
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pydantic import BaseModel
import yaml
from dataclasses import dataclass, field
import time
import random

@dataclass
class WorkflowStep:
    """Single step in an SEO workflow."""
    name: str
    description: str
    action: Callable
    dependencies: List[str] = field(default_factory=list)
    estimated_time: int = 0  # minutes
    user_input_required: bool = False
    customizable: bool = True
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow step."""
        return self.action(context)

class WorkflowTemplate(BaseModel):
    """Template for SEO workflows."""
    name: str
    description: str
    steps: List[str]
    estimated_duration: int  # minutes
    required_tools: List[str]
    customization_points: List[str]

class CustomizedWorkflow(BaseModel):
    """User-customized workflow."""
    base_template: str
    added_steps: List[str] = []
    removed_steps: List[str] = []
    modified_steps: Dict[str, Dict[str, Any]] = {}
    user_preferences: Dict[str, Any] = {}

class WorkflowEngine:
    """Engine for managing and executing SEO workflows."""
    
    def __init__(self):
        """Initialize workflow engine."""
        self.templates = self._load_default_templates()
        self.steps = self._load_default_steps()
        
        # Add default custom analysis step
        self.steps["custom_analysis"] = WorkflowStep(
            name="custom_analysis",
            description="Custom analysis step",
            action=lambda ctx: {"custom_analysis": "completed"},
            dependencies=[],
            estimated_time=15,
            user_input_required=True,
            customizable=True
        )
    
    def _load_default_templates(self) -> Dict[str, WorkflowTemplate]:
        """Load default workflow templates."""
        return {
            "technical_audit": WorkflowTemplate(
                name="Technical SEO Audit",
                description="Comprehensive technical SEO audit workflow",
                steps=[
                    "crawl_site",
                    "analyze_core_vitals",
                    "check_mobile_optimization",
                    "audit_site_structure",
                    "generate_technical_report"
                ],
                estimated_duration=180,
                required_tools=["crawler", "performance_analyzer"],
                customization_points=[
                    "crawl_depth",
                    "performance_metrics",
                    "mobile_checks"
                ]
            ),
            "content_audit": WorkflowTemplate(
                name="Content Audit",
                description="Content quality and performance audit workflow",
                steps=[
                    "inventory_content",
                    "analyze_performance",
                    "identify_gaps",
                    "generate_content_report"
                ],
                estimated_duration=120,
                required_tools=["content_analyzer", "gap_analyzer"],
                customization_points=[
                    "content_metrics",
                    "gap_analysis_depth"
                ]
            )
        }
    
    def _load_default_steps(self) -> Dict[str, WorkflowStep]:
        """Load default workflow steps."""
        return {
            "crawl_site": WorkflowStep(
                name="Crawl Site",
                description="Crawl website to collect technical data",
                action=self._crawl_site,
                estimated_time=60,
                customizable=True
            ),
            "analyze_core_vitals": WorkflowStep(
                name="Analyze Core Web Vitals",
                description="Analyze Core Web Vitals performance",
                action=self._analyze_core_vitals,
                dependencies=["crawl_site"],
                estimated_time=30,
                customizable=True
            ),
            "check_mobile_optimization": WorkflowStep(
                name="Check Mobile Optimization",
                description="Analyze mobile-friendliness",
                action=self._check_mobile_optimization,
                dependencies=["crawl_site"],
                estimated_time=30,
                customizable=True
            ),
            "audit_site_structure": WorkflowStep(
                name="Audit Site Structure",
                description="Analyze site structure and navigation",
                action=self._audit_site_structure,
                dependencies=["crawl_site"],
                estimated_time=45,
                customizable=True
            ),
            "generate_technical_report": WorkflowStep(
                name="Generate Technical Report",
                description="Generate technical SEO audit report",
                action=self._generate_technical_report,
                dependencies=["crawl_site", "analyze_core_vitals", "check_mobile_optimization", "audit_site_structure"],
                estimated_time=20,
                customizable=True
            )
        }
    
    def create_custom_workflow(
        self,
        base_template: str,
        customizations: Dict[str, Any]
    ) -> CustomizedWorkflow:
        """Create customized workflow from template.
        
        Args:
            base_template: Name of base template
            customizations: Customization options
            
        Returns:
            CustomizedWorkflow: Customized workflow
        """
        if base_template not in self.templates:
            raise ValueError(f"Unknown template: {base_template}")
        
        return CustomizedWorkflow(
            base_template=base_template,
            **customizations
        )
    
    def execute_workflow(
        self,
        workflow: CustomizedWorkflow,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow with given context.
        
        Args:
            workflow: Workflow to execute
            context: Execution context
            
        Returns:
            Dict[str, Any]: Workflow results
        """
        template = self.templates[workflow.base_template]
        steps = self._get_workflow_steps(workflow)
        
        results = {}
        for step in steps:
            if step.user_input_required:
                # Wait for user input
                user_input = self._get_user_input(step, context)
                context.update(user_input)
            
            step_result = step.execute(context)
            results[step.name] = step_result
            context.update(step_result)
        
        return results
    
    def _get_workflow_steps(
        self,
        workflow: CustomizedWorkflow
    ) -> List[WorkflowStep]:
        """Get ordered list of workflow steps."""
        template = self.templates[workflow.base_template]
        steps = []
        
        for step_name in template.steps:
            if step_name in workflow.removed_steps:
                continue
                
            step = self.steps[step_name]
            if step_name in workflow.modified_steps:
                # Apply user modifications
                modifications = workflow.modified_steps[step_name]
                step = self._modify_step(step, modifications)
            
            steps.append(step)
        
        # Add user-added steps
        for step_name in workflow.added_steps:
            steps.append(self.steps[step_name])
        
        return self._order_steps(steps)
    
    def _modify_step(
        self,
        step: WorkflowStep,
        modifications: Dict[str, Any]
    ) -> WorkflowStep:
        """Apply modifications to workflow step."""
        if not step.customizable:
            raise ValueError(f"Step {step.name} is not customizable")
            
        # Create new step with modifications
        return WorkflowStep(
            name=step.name,
            description=modifications.get("description", step.description),
            action=modifications.get("action", step.action),
            dependencies=modifications.get("dependencies", step.dependencies),
            estimated_time=modifications.get("estimated_time", step.estimated_time),
            user_input_required=modifications.get(
                "user_input_required",
                step.user_input_required
            ),
            customizable=step.customizable
        )
    
    def _order_steps(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Order steps based on dependencies."""
        # Create a dict of step name to step
        step_dict = {step.name: step for step in steps}
        
        # Create a dict of step name to dependencies
        dependencies = {step.name: set(step.dependencies) for step in steps}
        
        # Find steps with no dependencies
        no_deps = [name for name, deps in dependencies.items() if not deps]
        
        # Topologically sort steps
        ordered = []
        while no_deps:
            # Take a step with no dependencies
            step_name = no_deps.pop(0)
            ordered.append(step_dict[step_name])
            
            # Remove this step from other steps' dependencies
            for name, deps in dependencies.items():
                if step_name in deps:
                    deps.remove(step_name)
                    if not deps:
                        no_deps.append(name)
        
        # Check for circular dependencies
        if len(ordered) != len(steps):
            raise ValueError("Circular dependencies detected in workflow steps")
        
        return ordered

    def _get_user_input(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get required user input for step."""
        # Implementation would handle user interaction
        return {}
    
    def _crawl_site(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Crawl website to collect technical data."""
        # Simulate crawling
        time.sleep(0.1)
        return {
            "pages_crawled": len(context.get("pages", [])),
            "status_codes": {
                "200": random.randint(90, 100),
                "404": random.randint(0, 5),
                "500": random.randint(0, 2)
            }
        }

    def _analyze_core_vitals(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Core Web Vitals performance."""
        # Simulate analysis
        time.sleep(0.1)
        return {
            "lcp": random.uniform(1.5, 4.0),
            "fid": random.uniform(50, 200),
            "cls": random.uniform(0.05, 0.25)
        }

    def _check_mobile_optimization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check mobile optimization."""
        # Simulate mobile optimization check
        time.sleep(0.1)
        return {
            "mobile_friendly": random.random() > 0.2,
            "viewport_configured": random.random() > 0.1,
            "text_readable": random.random() > 0.15,
            "tap_targets": {
                "too_small": random.randint(0, 10),
                "too_close": random.randint(0, 5)
            }
        }

    def _audit_site_structure(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Audit site structure."""
        # Simulate site structure analysis
        time.sleep(0.1)
        return {
            "depth": random.randint(1, 5),
            "orphaned_pages": random.randint(0, 10),
            "broken_links": random.randint(0, 5),
            "navigation": {
                "menu_depth": random.randint(1, 3),
                "breadcrumbs": random.random() > 0.2,
                "sitemap": random.random() > 0.1
            }
        }

    def _generate_technical_report(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical SEO report."""
        # Simulate report generation
        time.sleep(0.1)
        return {
            "summary": {
                "score": random.randint(60, 100),
                "critical_issues": random.randint(0, 5),
                "warnings": random.randint(0, 10),
                "suggestions": random.randint(0, 15)
            },
            "sections": {
                "core_vitals": context.get("core_vitals", {}),
                "mobile": context.get("mobile_optimization", {}),
                "structure": context.get("site_structure", {}),
                "recommendations": [
                    "Optimize images",
                    "Improve mobile experience",
                    "Fix broken links"
                ]
            }
        }
