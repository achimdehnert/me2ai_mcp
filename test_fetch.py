"""Test script for directly testing the Fetch MCP functionality."""
import asyncio
from fetch_mcp import FetchMCPServer

async def main():
    """Run a simple test of the Fetch MCP server functionality."""
    print("Initializing Fetch MCP Server...")
    server = FetchMCPServer()
    
    print("\n=== Testing fetch_webpage ===")
    url = "https://www.hnu.de/"
    result = await server.fetch_webpage(url)
    if result["success"]:
        print(f"Successfully fetched: {url}")
        print(f"Title: {result['title']}")
        print(f"Content length: {result['content_length']} characters")
        print("\nFirst 500 characters of content:")
        print(result["content"][:500] + "...")
    else:
        print(f"Error fetching webpage: {result.get('error')}")
    
    print("\n=== Testing summarize_webpage ===")
    result = await server.summarize_webpage(url)
    if result["success"]:
        print(f"Successfully summarized: {url}")
        print(f"Title: {result['title']}")
        if result.get("headings"):
            print("\nMain headings:")
            for heading in result["headings"][:5]:
                print(f"- {heading['text']}")
        
        if result.get("main_paragraphs"):
            print("\nKey content (first paragraph):")
            print(result["main_paragraphs"][0][:200] + "...")
    else:
        print(f"Error summarizing webpage: {result.get('error')}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    asyncio.run(main())
