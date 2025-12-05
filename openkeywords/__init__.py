"""
OpenKeywords - AI-powered SEO keyword generation using Gemini + SE Ranking + Deep Research.

Generate high-quality, clustered SEO keywords for any business.

Features:
- AI keyword generation (Gemini)
- SE Ranking gap analysis (competitor keywords)
- Deep Research (Reddit, Quora, forums) for hyper-niche keywords

Usage:
    from openkeywords import KeywordGenerator, CompanyInfo, GenerationConfig

    generator = KeywordGenerator()
    result = await generator.generate(
        CompanyInfo(name="Acme Software", industry="B2B SaaS"),
        GenerationConfig(enable_research=True),  # Enable deep research
    )

    for kw in result.keywords:
        print(f"{kw.keyword} | {kw.intent} | Score: {kw.score} | Source: {kw.source}")
"""

from .models import (
    Cluster,
    CompanyInfo,
    GenerationConfig,
    GenerationResult,
    Keyword,
    KeywordStatistics,
)
from .generator import KeywordGenerator
from .seranking_client import SEORankingAPIClient
from .gap_analyzer import SEORankingAPI, AEOContentGapAnalyzer
from .researcher import ResearchEngine

__version__ = "0.2.0"
__all__ = [
    # Main API
    "KeywordGenerator",
    "CompanyInfo",
    "GenerationConfig",
    "GenerationResult",
    "Keyword",
    "Cluster",
    "KeywordStatistics",
    # SE Ranking
    "SEORankingAPIClient",
    "SEORankingAPI",
    "AEOContentGapAnalyzer",
    # Deep Research
    "ResearchEngine",
]
