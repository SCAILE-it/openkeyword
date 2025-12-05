"""
OpenKeywords - Deep Research Example

Deep Research uses Google Search grounding to find hyper-niche keywords
from Reddit, Quora, forums, and other community discussions.

Before running, set your API key:
    export GEMINI_API_KEY='your-gemini-api-key'

Run from project root:
    python examples/with_research.py
"""

import asyncio
import os

from openkeywords import KeywordGenerator, CompanyInfo, GenerationConfig


async def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: Set GEMINI_API_KEY environment variable")
        return

    generator = KeywordGenerator()

    # Define company
    company = CompanyInfo(
        name="DevOps Pro",
        industry="DevOps & Cloud Infrastructure",
        description="CI/CD pipeline and cloud deployment solutions",
        services=["CI/CD automation", "Kubernetes management", "cloud migration"],
        products=["DevOps Pro Platform", "Pipeline Builder"],
        target_audience="software development teams",
        target_location="United States",
    )

    # Enable deep research to find hyper-niche keywords
    config = GenerationConfig(
        target_count=40,
        min_score=40,
        enable_clustering=True,
        cluster_count=5,
        language="english",
        region="us",
        enable_research=True,  # This enables Reddit, Quora, forum search
    )

    print(f"Deep Research Demo")
    print(f"Company: {company.name}")
    print(f"Research: ENABLED (Reddit, Quora, forums)")

    result = await generator.generate(company, config)

    print(f"Generated {len(result.keywords)} keywords in {result.processing_time_seconds:.1f}s")

    # Show source breakdown
    print("Keyword Sources:")
    for source, count in result.statistics.source_breakdown.items():
        pct = (count / len(result.keywords)) * 100
        print(f"  {source}: {count} ({pct:.0f}%)")

    # Show research-found keywords specifically
    research_keywords = [kw for kw in result.keywords if "research" in kw.source]
    if research_keywords:
        print(f"Keywords from Deep Research ({len(research_keywords)}):")
        for kw in research_keywords[:15]:
            print(f"  [{kw.source}] {kw.keyword}")

    # Export with source info
    result.to_csv("keywords_with_research.csv")
    result.to_json("keywords_with_research.json")


if __name__ == "__main__":
    asyncio.run(main())
