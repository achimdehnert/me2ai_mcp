"""SEO reporting templates and generators."""
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import json
import yaml

class ReportSection(BaseModel):
    """Section of an SEO report."""
    title: str
    metrics: Dict[str, Any]
    insights: List[str]
    recommendations: List[Dict[str, Any]]
    priority: str = "medium"
    
    def __init__(self, **data):
        """Initialize report section."""
        super().__init__(**data)
        if self.priority not in ["low", "medium", "high"]:
            raise ValueError("Priority must be one of: low, medium, high")

class ReportTemplate(BaseModel):
    """Template for SEO reports."""
    name: str
    sections: List[str]
    metrics: List[str]
    visualizations: List[str]
    custom_fields: Optional[Dict[str, Any]] = None
    
    def __init__(self, **data):
        """Initialize report template."""
        super().__init__(**data)
        if not self.name:
            raise ValueError("Template name cannot be empty")
        if not self.sections:
            raise ValueError("Template must have at least one section")

class CustomReport(BaseModel):
    """User-customized report template."""
    base_template: str
    added_sections: List[str] = []
    removed_sections: List[str] = []
    custom_metrics: Dict[str, Any] = {}
    
    def to_template(self) -> ReportTemplate:
        """Convert custom report to template."""
        base = REPORT_TEMPLATES[self.base_template]
        sections = [s for s in base.sections if s not in self.removed_sections]
        sections.extend(self.added_sections)
        
        # Convert base metrics list to set and add custom metrics
        metrics = set(base.metrics)
        metrics.update(self.custom_metrics.keys())
        
        return ReportTemplate(
            name=f"Custom_{base.name}",
            sections=sections,
            metrics=list(metrics),
            visualizations=base.visualizations
        )

# Standard report templates
REPORT_TEMPLATES = {
    "technical_audit": ReportTemplate(
        name="Technical SEO Audit",
        sections=[
            "Core Web Vitals",
            "Mobile Optimization",
            "Site Architecture",
            "Security & SSL",
            "Schema Implementation"
        ],
        metrics=[
            "page_speed_metrics",
            "mobile_usability",
            "crawl_efficiency",
            "security_score",
            "schema_coverage"
        ],
        visualizations=[
            "performance_trends",
            "mobile_desktop_comparison",
            "crawl_budget_analysis"
        ]
    ),
    "content_performance": ReportTemplate(
        name="Content Performance Analysis",
        sections=[
            "Top Performing Content",
            "Content Gaps",
            "Keyword Rankings",
            "User Engagement",
            "Conversion Analysis"
        ],
        metrics=[
            "organic_traffic",
            "engagement_metrics",
            "conversion_rates",
            "keyword_positions",
            "content_roi"
        ],
        visualizations=[
            "traffic_trends",
            "content_performance_matrix",
            "keyword_position_distribution"
        ]
    ),
    "competitive_analysis": ReportTemplate(
        name="Competitive Analysis",
        sections=[
            "Market Share",
            "Keyword Gap",
            "Backlink Profile",
            "Content Comparison",
            "Technical Comparison"
        ],
        metrics=[
            "market_share_metrics",
            "keyword_overlap",
            "backlink_metrics",
            "content_metrics",
            "technical_scores"
        ],
        visualizations=[
            "market_share_pie",
            "keyword_gap_analysis",
            "backlink_comparison"
        ]
    )
}

def load_custom_template(template_path: str) -> ReportTemplate:
    """Load custom report template from file.
    
    Args:
        template_path: Path to template file (JSON/YAML)
        
    Returns:
        ReportTemplate: Loaded template
    """
    with open(template_path, 'r') as f:
        if template_path.endswith('.json'):
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    return ReportTemplate(**data)

def save_custom_template(template: ReportTemplate, output_path: str):
    """Save custom report template to file.
    
    Args:
        template: Template to save
        output_path: Path to save template to
    """
    data = template.dict()
    with open(output_path, 'w') as f:
        if output_path.endswith('.json'):
            json.dump(data, f, indent=2)
        else:
            yaml.dump(data, f)
