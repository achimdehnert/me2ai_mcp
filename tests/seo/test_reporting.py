"""Tests for SEO reporting functionality."""
import pytest
from datetime import datetime
from me2ai.tools.seo.reporting.templates import (
    ReportTemplate,
    CustomReport,
    ReportSection,
    REPORT_TEMPLATES,
    load_custom_template,
    save_custom_template
)

@pytest.fixture
def sample_template():
    """Sample report template for testing."""
    return ReportTemplate(
        name="Test Template",
        sections=["Section1", "Section2"],
        metrics=["metric1", "metric2"],
        visualizations=["viz1", "viz2"]
    )

@pytest.fixture
def sample_section():
    """Sample report section for testing."""
    return ReportSection(
        title="Test Section",
        metrics={"metric1": 10, "metric2": 20},
        insights=["Insight 1", "Insight 2"],
        recommendations=[
            {"action": "Action 1", "priority": "high"},
            {"action": "Action 2", "priority": "medium"}
        ]
    )

class TestReportTemplate:
    """Test report template functionality."""
    
    def test_template_creation(self, sample_template):
        """Test creating a report template."""
        assert sample_template.name == "Test Template"
        assert len(sample_template.sections) == 2
        assert len(sample_template.metrics) == 2
        assert len(sample_template.visualizations) == 2
    
    def test_template_validation(self):
        """Test template validation."""
        with pytest.raises(ValueError):
            ReportTemplate(
                name="",  # Empty name should fail
                sections=[],
                metrics=[],
                visualizations=[]
            )
    
    def test_custom_fields(self, sample_template):
        """Test adding custom fields to template."""
        sample_template.custom_fields = {
            "field1": "value1",
            "field2": "value2"
        }
        assert len(sample_template.custom_fields) == 2
        assert sample_template.custom_fields["field1"] == "value1"

class TestCustomReport:
    """Test custom report functionality."""
    
    def test_custom_report_creation(self):
        """Test creating a custom report."""
        custom_report = CustomReport(
            base_template="technical_audit",
            added_sections=["New Section"],
            removed_sections=["Core Web Vitals"],
            custom_metrics={"new_metric": "value"}
        )
        assert custom_report.base_template == "technical_audit"
        assert len(custom_report.added_sections) == 1
        assert len(custom_report.removed_sections) == 1
    
    def test_template_conversion(self):
        """Test converting custom report to template."""
        custom_report = CustomReport(
            base_template="technical_audit",
            added_sections=["New Section"]
        )
        template = custom_report.to_template()
        assert "New Section" in template.sections
    
    def test_invalid_base_template(self):
        """Test using invalid base template."""
        with pytest.raises(KeyError):
            CustomReport(
                base_template="nonexistent_template"
            ).to_template()

class TestReportSection:
    """Test report section functionality."""
    
    def test_section_creation(self, sample_section):
        """Test creating a report section."""
        assert sample_section.title == "Test Section"
        assert len(sample_section.metrics) == 2
        assert len(sample_section.insights) == 2
        assert len(sample_section.recommendations) == 2
    
    def test_section_priority(self, sample_section):
        """Test section priority handling."""
        assert sample_section.priority == "medium"
        sample_section.priority = "high"
        assert sample_section.priority == "high"
        
        with pytest.raises(ValueError):
            sample_section.priority = "invalid_priority"

class TestTemplateIO:
    """Test template I/O operations."""
    
    def test_save_load_json(self, sample_template, tmp_path):
        """Test saving and loading template as JSON."""
        file_path = tmp_path / "template.json"
        save_custom_template(sample_template, str(file_path))
        loaded = load_custom_template(str(file_path))
        assert loaded.name == sample_template.name
        assert loaded.sections == sample_template.sections
    
    def test_save_load_yaml(self, sample_template, tmp_path):
        """Test saving and loading template as YAML."""
        file_path = tmp_path / "template.yaml"
        save_custom_template(sample_template, str(file_path))
        loaded = load_custom_template(str(file_path))
        assert loaded.name == sample_template.name
        assert loaded.sections == sample_template.sections
    
    def test_invalid_file_format(self, sample_template, tmp_path):
        """Test handling invalid file format."""
        file_path = tmp_path / "template.invalid"
        with pytest.raises(ValueError):
            save_custom_template(sample_template, str(file_path))

def test_default_templates():
    """Test default template configurations."""
    assert "technical_audit" in REPORT_TEMPLATES
    assert "content_performance" in REPORT_TEMPLATES
    assert "competitive_analysis" in REPORT_TEMPLATES
    
    technical = REPORT_TEMPLATES["technical_audit"]
    assert "Core Web Vitals" in technical.sections
    assert "mobile_usability" in technical.metrics
