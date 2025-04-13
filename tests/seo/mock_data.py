"""Mock data generators for SEO testing."""
import random
from typing import Dict, List, Any
from datetime import datetime, timedelta
import numpy as np
from faker import Faker
from faker.providers import company, internet, date_time, lorem

fake = Faker()
fake.add_provider(company)
fake.add_provider(internet)
fake.add_provider(date_time)
fake.add_provider(lorem)

class SEOMockData:
    """Generator for SEO-related mock data."""
    
    def __init__(self, seed: int = None):
        """Initialize mock data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            fake.seed_instance(seed)
    
    def generate_page_metrics(self) -> Dict[str, Any]:
        """Generate mock page performance metrics."""
        return {
            "load_time": round(random.uniform(1.0, 5.0), 2),
            "fcp": round(random.uniform(0.8, 3.0), 2),
            "lcp": round(random.uniform(1.5, 4.0), 2),
            "cls": round(random.uniform(0.0, 0.5), 3),
            "fid": round(random.uniform(50, 200), 0),
            "mobile_friendly": random.random() > 0.1,
            "ssl_valid": random.random() > 0.05
        }
    
    def generate_keyword_data(self, num_keywords: int = 10) -> List[Dict[str, Any]]:
        """Generate mock keyword data."""
        if num_keywords < 0:
            raise ValueError("Number of keywords must be non-negative")
        return [
            {
                "keyword": fake.words(nb=random.randint(1, 4)),
                "search_volume": random.randint(100, 10000),
                "difficulty": random.randint(1, 100),
                "current_rank": random.randint(1, 100) if random.random() > 0.3 else None,
                "cpc": round(random.uniform(0.1, 10.0), 2),
                "competition": random.uniform(0, 1),
                "intent": random.choice(["informational", "navigational", "transactional", "commercial"])
            }
            for _ in range(num_keywords)
        ]
    
    def generate_traffic_data(
        self,
        days: int = 30,
        trend: str = "random"
    ) -> List[Dict[str, Any]]:
        """Generate mock traffic data with trends."""
        if days < 0:
            raise ValueError("Number of days must be non-negative")
        if trend not in ["random", "up", "down", "seasonal"]:
            raise KeyError(f"Invalid trend type: {trend}. Must be one of: random, up, down, seasonal")
            
        base_traffic = random.randint(1000, 5000)
        trend_factor = {
            "up": lambda x: x * 1.02,
            "down": lambda x: x * 0.98,
            "seasonal": lambda x: x + np.sin(x / 7) * 500,
            "random": lambda x: x * random.uniform(0.95, 1.05)
        }[trend]
        
        data = []
        current_traffic = base_traffic
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i)
            current_traffic = trend_factor(current_traffic)
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "visitors": max(int(current_traffic), 0),  # Ensure non-negative
                "pageviews": max(int(current_traffic * random.uniform(1.5, 2.5)), 0),
                "bounce_rate": random.uniform(0.3, 0.7),
                "avg_time": random.uniform(60, 300)
            })
        
        return data
    
    def generate_backlink_data(self, num_links: int = 50) -> List[Dict[str, Any]]:
        """Generate mock backlink data."""
        if num_links < 0:
            raise ValueError("Number of backlinks must be non-negative")
        return [
            {
                "url": fake.url(),
                "domain_authority": random.randint(1, 100),
                "spam_score": random.randint(1, 100),
                "anchor_text": fake.words(nb=random.randint(1, 5)),
                "dofollow": random.random() > 0.2,
                "first_seen": fake.date_this_year(),
                "last_seen": fake.date_this_month()
            }
            for _ in range(num_links)
        ]
    
    def generate_content_data(self, num_pages: int = 20) -> List[Dict[str, Any]]:
        """Generate mock content performance data."""
        if num_pages < 0:
            raise ValueError("Number of pages must be non-negative")
        return [
            {
                "url": f"/page-{i}",
                "title": fake.sentence(),
                "word_count": random.randint(300, 3000),
                "readability_score": random.randint(50, 100),
                "publish_date": fake.date_this_year(),
                "last_updated": fake.date_this_month(),
                "performance": {
                    "organic_traffic": random.randint(100, 1000),
                    "conversion_rate": random.uniform(0.01, 0.05),
                    "bounce_rate": random.uniform(0.3, 0.7),
                    "avg_time": random.uniform(60, 300)
                }
            }
            for i in range(num_pages)
        ]
    
    def generate_competitor_data(self, num_competitors: int = 5) -> List[Dict[str, Any]]:
        """Generate mock competitor data."""
        if num_competitors < 0:
            raise ValueError("Number of competitors must be non-negative")
        return [
            {
                "domain": fake.domain_name(),
                "metrics": {
                    "domain_authority": random.randint(1, 100),
                    "organic_keywords": random.randint(1000, 10000),
                    "organic_traffic": random.randint(10000, 100000),
                    "backlinks": random.randint(1000, 10000)
                },
                "overlap": {
                    "keywords": random.randint(100, 1000),
                    "percentage": random.uniform(0.1, 0.5)
                },
                "growth": random.uniform(-0.2, 0.4)
            }
            for _ in range(num_competitors)
        ]
    
    def generate_local_seo_data(self) -> Dict[str, Any]:
        """Generate mock local SEO data."""
        return {
            "gmb": {
                "name": fake.company(),
                "address": fake.address(),
                "phone": fake.phone_number(),
                "category": random.choice([
                    "Restaurant", "Retail", "Professional Services",
                    "Healthcare", "Automotive"
                ]),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "reviews": random.randint(10, 500),
                "photos": random.randint(5, 50)
            },
            "citations": [
                {
                    "site": site,
                    "status": random.choice(["active", "pending", "incomplete"]),
                    "accuracy": random.uniform(0.8, 1.0)
                }
                for site in ["Yelp", "YellowPages", "BBB", "Facebook", "Bing"]
            ],
            "local_pack": {
                "position": random.randint(1, 3),
                "visibility": random.uniform(0.3, 0.9),
                "impression_share": random.uniform(0.2, 0.8)
            }
        }
    
    def generate_ecommerce_data(self, num_products: int = 50) -> Dict[str, Any]:
        """Generate mock e-commerce SEO data."""
        if num_products < 0:
            raise ValueError("Number of products must be non-negative")
        return {
            "products": [
                {
                    "sku": fake.ean13(),
                    "name": fake.catch_phrase(),
                    "category": random.choice([
                        "Electronics", "Clothing", "Home", "Sports", "Books"
                    ]),
                    "price": round(random.uniform(10, 1000), 2),
                    "inventory": random.randint(0, 100),
                    "performance": {
                        "impressions": random.randint(100, 1000),
                        "clicks": random.randint(10, 100),
                        "conversions": random.randint(1, 10),
                        "revenue": round(random.uniform(100, 1000), 2)
                    }
                }
                for _ in range(num_products)
            ],
            "categories": [
                {
                    "name": category,
                    "products": random.randint(10, 100),
                    "depth": random.randint(1, 3),
                    "performance": {
                        "traffic": random.randint(1000, 5000),
                        "conversion_rate": random.uniform(0.01, 0.05)
                    }
                }
                for category in ["Electronics", "Clothing", "Home", "Sports", "Books"]
            ]
        }
