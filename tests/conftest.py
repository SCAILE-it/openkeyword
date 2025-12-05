"""
Shared fixtures for OpenKeywords tests
"""

import pytest
from openkeywords import CompanyInfo, GenerationConfig, Keyword, Cluster, GenerationResult


@pytest.fixture
def sample_company():
    """Sample company info for testing"""
    return CompanyInfo(
        name="TestCorp",
        url="https://testcorp.com",
        industry="B2B SaaS",
        description="Project management software for teams",
        services=["project management", "team collaboration"],
        products=["TestCorp Pro", "TestCorp Teams"],
        target_audience="small businesses",
        target_location="United States",
        competitors=["competitor1.com", "competitor2.com"],
    )


@pytest.fixture
def sample_config():
    """Sample generation config"""
    return GenerationConfig(
        target_count=20,
        min_score=40,
        enable_clustering=True,
        cluster_count=4,
        language="english",
        region="us",
    )


@pytest.fixture
def sample_keywords():
    """Sample keyword dicts (internal format)"""
    return [
        {"keyword": "best project management software", "intent": "commercial", "score": 85, "is_question": False},
        {"keyword": "how to manage projects remotely", "intent": "question", "score": 75, "is_question": True},
        {"keyword": "project management tools pricing", "intent": "commercial", "score": 80, "is_question": False},
        {"keyword": "sign up for project management", "intent": "transactional", "score": 70, "is_question": False},
        {"keyword": "team collaboration software", "intent": "informational", "score": 72, "is_question": False},
    ]


@pytest.fixture
def sample_keywords_with_duplicates():
    """Keywords with duplicates for dedup testing"""
    return [
        {"keyword": "best project management software", "intent": "commercial", "score": 85},
        {"keyword": "best software project management", "intent": "commercial", "score": 75},
        {"keyword": "Best Project Management Software", "intent": "commercial", "score": 80},
        {"keyword": "sign up for project management", "intent": "transactional", "score": 70},
        {"keyword": "sign up project management", "intent": "transactional", "score": 65},
        {"keyword": "project management pricing", "intent": "commercial", "score": 60},
    ]


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response"""
    return {
        "keywords": [
            {"keyword": "best project management software", "intent": "commercial", "is_question": False},
            {"keyword": "how to manage remote teams", "intent": "question", "is_question": True},
            {"keyword": "project management pricing", "intent": "commercial", "is_question": False},
        ]
    }
