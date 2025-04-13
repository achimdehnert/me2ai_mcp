"""Direct webpage fetch with output to file."""
import requests
from bs4 import BeautifulSoup
import sys

def fetch_hnu():
    """Fetch the HNU website and output a summary."""
    url = "https://www.hnu.de/"
    
    try:
        # Request the webpage
        print(f"Fetching {url}...")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get basic information
        title = soup.title.string if soup.title else "No title"
        print(f"\nTitle: {title}")
        
        # Get headings
        headings = []
        for tag in ['h1', 'h2', 'h3']:
            for heading in soup.find_all(tag):
                text = heading.get_text(strip=True)
                if text and len(text) > 3:  # Skip very short headings
                    headings.append(f"{tag}: {text}")
        
        print("\nTop Headings:")
        for heading in headings[:10]:
            print(f"- {heading}")
        
        # Get main content paragraphs
        print("\nMain Content Excerpts:")
        paragraphs = soup.find_all('p')
        content_paragraphs = []
        
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and len(text) > 50:  # Only substantial paragraphs
                content_paragraphs.append(text)
        
        for i, text in enumerate(content_paragraphs[:3], 1):
            if len(text) > 200:
                text = text[:200] + "..."
            print(f"{i}. {text}")
        
        print("\nFetch completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return False

if __name__ == "__main__":
    # Redirect output to both console and file
    orig_stdout = sys.stdout
    with open('hnu_summary.txt', 'w', encoding='utf-8') as f:
        class MultiWriter:
            def write(self, x):
                orig_stdout.write(x)
                f.write(x)
            def flush(self):
                orig_stdout.flush()
                f.flush()
        
        sys.stdout = MultiWriter()
        fetch_hnu()
        sys.stdout = orig_stdout
    
    print("Fetch complete. Results saved to hnu_summary.txt")
