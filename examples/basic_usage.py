"""
OpenKeywords - Basic Usage Example

Before running, set your API key:
    export GEMINI_API_KEY='your-gemini-api-key'

Run from project root:
    python examples/basic_usage.py
"""

import asyncio
import os

from openkeywords import KeywordGenerator, CompanyInfo, GenerationConfig


async def main():
    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: Set GEMINI_API_KEY environment variable")
        print("  export GEMINI_API_KEY='your-key'")
        return

    # Initialize generator (uses GEMINI_API_KEY env var)
    generator = KeywordGenerator()

    # Define company information
    company = CompanyInfo(
        name="TechStartup.io",
        industry="B2B SaaS",
        description="Project management software for remote teams",
        services=["project management", "team collaboration", "task tracking"],
        products=["TechStartup Pro", "TechStartup Teams"],
        target_audience="small to medium businesses",
        target_location="United States",
    )

    # Configure generation
    config = GenerationConfig(
        target_count=30,
        min_score=50,
        enable_clustering=True,
        cluster_count=5,
        language="english",
        region="us",
    )

    print(f"Generating keywords for {company.name}...")

    # Generate keywords
    result = await generator.generate(company, config)

    # Display results
    print(f"Generated {len(result.keywords)} keywords in {result.processing_time_seconds:.1f}s")

    for kw in result.keywords[:10]:
        print(f"{kw.keyword} | {kw.intent} | Score: {kw.score}")

    # Export to files
    result.to_csv("keywords_output.csv")
    result.to_json("keywords_output.json")


if __name__ == "__main__":
    asyncio.run(main())
