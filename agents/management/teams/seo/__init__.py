"""SEO team package."""

from .general import SEOTeam
from .ecommerce import create_ecommerce_seo_team
from .local import create_local_seo_team

__all__ = [
    'SEOTeam',
    'create_ecommerce_seo_team',
    'create_local_seo_team'
]
