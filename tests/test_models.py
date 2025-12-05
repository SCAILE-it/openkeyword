"""
Tests for OpenKeywords data models
"""

import pytest
import tempfile
import json
import csv
from pathlib import Path

from openkeywords import (
    CompanyInfo,
    GenerationConfig,
    Keyword,
    Cluster,
    GenerationResult,
    KeywordStatistics,
)


class TestCompanyInfo:
    """Tests for CompanyInfo model"""

    def test_minimal_company(self):
        """Test company with only required fields"""
        company = CompanyInfo(name="TestCorp")
        assert company.name == "TestCorp"
        assert company.url == ""
        assert company.services == []

    def test_full_company(self, sample_company):
        """Test company with all fields"""
        assert sample_company.name == "TestCorp"
        assert sample_company.industry == "B2B SaaS"
        assert len(sample_company.services) == 2
        assert len(sample_company.competitors) == 2


class TestGenerationConfig:
    """Tests for GenerationConfig model"""

    def test_default_config(self):
        """Test default configuration values"""
        config = GenerationConfig()
        assert config.target_count == 50
        assert config.min_score == 40
        assert config.enable_clustering is True
        assert config.cluster_count == 6
        assert config.language == "english"
        assert config.region == "us"

    def test_custom_config(self):
        """Test custom configuration"""
        config = GenerationConfig(
            target_count=100,
            min_score=60,
            language="german",
            region="de",
        )
        assert config.target_count == 100
        assert config.min_score == 60
        assert config.language == "german"
        assert config.region == "de"


class TestKeyword:
    """Tests for Keyword model"""

    def test_keyword_defaults(self):
        """Test keyword with defaults"""
        kw = Keyword(keyword="test keyword")
        assert kw.keyword == "test keyword"
        assert kw.intent == "informational"
        assert kw.score == 0
        assert kw.is_question is False
        assert kw.volume == 0
        assert kw.difficulty == 50

    def test_keyword_full(self):
        """Test keyword with all fields"""
        kw = Keyword(
            keyword="how to test software",
            intent="question",
            score=85,
            cluster_name="Testing",
            is_question=True,
            volume=1200,
            difficulty=35,
        )
        assert kw.keyword == "how to test software"
        assert kw.intent == "question"
        assert kw.score == 85
        assert kw.is_question is True
        assert kw.volume == 1200


class TestCluster:
    """Tests for Cluster model"""

    def test_cluster_count(self):
        """Test cluster keyword count property"""
        cluster = Cluster(
            name="Product Features",
            keywords=["feature 1", "feature 2", "feature 3"],
        )
        assert cluster.name == "Product Features"
        assert cluster.count == 3

    def test_empty_cluster(self):
        """Test empty cluster"""
        cluster = Cluster(name="Empty")
        assert cluster.count == 0


class TestGenerationResult:
    """Tests for GenerationResult model"""

    def test_result_to_csv(self):
        """Test CSV export"""
        result = GenerationResult(
            keywords=[
                Keyword(keyword="test 1", intent="commercial", score=80),
                Keyword(keyword="test 2", intent="question", score=75),
            ],
            clusters=[Cluster(name="Test", keywords=["test 1", "test 2"])],
            statistics=KeywordStatistics(total=2, avg_score=77.5),
            processing_time_seconds=5.0,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name

        result.to_csv(filepath)

        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["keyword"] == "test 1"
        assert rows[0]["intent"] == "commercial"
        assert rows[1]["keyword"] == "test 2"

        Path(filepath).unlink()

    def test_result_to_json(self):
        """Test JSON export"""
        result = GenerationResult(
            keywords=[
                Keyword(keyword="test keyword", intent="commercial", score=80),
            ],
            clusters=[],
            statistics=KeywordStatistics(total=1),
            processing_time_seconds=2.0,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        result.to_json(filepath)

        with open(filepath) as f:
            data = json.load(f)

        assert len(data["keywords"]) == 1
        assert data["keywords"][0]["keyword"] == "test keyword"
        assert data["processing_time_seconds"] == 2.0

        Path(filepath).unlink()

    def test_result_to_dict(self):
        """Test dict conversion"""
        result = GenerationResult(
            keywords=[Keyword(keyword="test", score=50)],
            clusters=[],
            statistics=KeywordStatistics(total=1),
            processing_time_seconds=1.0,
        )

        data = result.to_dict()
        assert isinstance(data, dict)
        assert "keywords" in data
        assert "statistics" in data
