"""Simple script to fetch and display content from a webpage."""
import requests
from bs4 import BeautifulSoup

def fetch_website(url):
    """Fetch and analyze a website."""
    print(f"Fetching: {url}")
    
    try:
        # Fetch the webpage
        headers = {
            "User-Agent": "Mozilla/5.0 ME2AI Web Fetcher/1.0"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get the title
        title = soup.title.string if soup.title else "No title found"
        print(f"\nTitle: {title}")
        
        # Get meta description
        meta_description = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and "content" in meta_tag.attrs:
            meta_description = meta_tag["content"]
            print(f"\nDescription: {meta_description}")
        
        # Find headings
        print("\nMain Headings:")
        for i, heading in enumerate(soup.find_all(['h1', 'h2', 'h3'])[:10], 1):
            text = heading.get_text(strip=True)
            if text:
                print(f"{i}. {text}")
        
        # Find main content area (this is a heuristic approach)
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        
        if not main_content:
            # If no obvious content container, look for largest text block
            paragraphs = soup.find_all('p')
            if paragraphs:
                main_content = max(paragraphs, key=lambda p: len(p.get_text(strip=True)))
        
        # Extract some content
        print("\nSample Content:")
        if main_content:
            content = main_content.get_text(strip=True)
            print(content[:500] + "..." if len(content) > 500 else content)
        else:
            # Fall back to body if no main content identified
            body_text = soup.body.get_text(strip=True)
            print(body_text[:500] + "..." if len(body_text) > 500 else body_text)
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Test with HNU website
    fetch_website("https://www.hnu.de/")
