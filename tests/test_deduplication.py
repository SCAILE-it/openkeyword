"""
Tests for keyword deduplication logic
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

from openkeywords import KeywordGenerator


class TestFastDeduplication:
    """Tests for fast O(n) deduplication"""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked Gemini"""
        with patch("openkeywords.generator.genai") as mock_genai:
            mock_genai.configure = MagicMock()
            mock_genai.GenerativeModel = MagicMock()
            gen = KeywordGenerator(gemini_api_key="fake-key")
            return gen

    def test_exact_duplicate_removal(self, generator):
        """Test exact duplicate removal (case-insensitive)"""
        keywords = [
            {"keyword": "project management software", "score": 80},
            {"keyword": "Project Management Software", "score": 75},
            {"keyword": "team collaboration tools", "score": 70},
        ]

        result, dup_count = generator._deduplicate_fast(keywords)

        assert len(result) == 2
        assert dup_count == 1
        assert result[0]["keyword"] == "project management software"

    def test_token_signature_dedup(self, generator):
        """Test token signature grouping (same words, different order)"""
        keywords = [
            {"keyword": "best project management", "score": 85},
            {"keyword": "project management best", "score": 75},
            {"keyword": "management project best", "score": 70},
            {"keyword": "team collaboration", "score": 65},
        ]

        result, dup_count = generator._deduplicate_fast(keywords)

        assert len(result) == 2
        assert dup_count == 2
        assert any(kw["keyword"] == "best project management" for kw in result)

    def test_gap_analysis_source_preferred(self, generator):
        """Test that gap_analysis source keywords are preferred"""
        keywords = [
            {"keyword": "software project management", "score": 70, "source": "ai_generated"},
            {"keyword": "project management software", "score": 85, "source": "gap_analysis"},
        ]

        result, dup_count = generator._deduplicate_fast(keywords)

        assert len(result) == 1
        assert result[0]["source"] == "gap_analysis"

    def test_empty_keywords(self, generator):
        """Test with empty input"""
        result, dup_count = generator._deduplicate_fast([])

        assert result == []
        assert dup_count == 0

    def test_no_duplicates(self, generator):
        """Test with no duplicates"""
        keywords = [
            {"keyword": "project management", "score": 80},
            {"keyword": "team collaboration", "score": 75},
            {"keyword": "task tracking", "score": 70},
        ]

        result, dup_count = generator._deduplicate_fast(keywords)

        assert len(result) == 3
        assert dup_count == 0

    def test_blank_keywords_filtered(self, generator):
        """Test that blank keywords are filtered out"""
        keywords = [
            {"keyword": "project management", "score": 80},
            {"keyword": "", "score": 75},
            {"keyword": "   ", "score": 70},
            {"keyword": "team collaboration", "score": 65},
        ]

        result, dup_count = generator._deduplicate_fast(keywords)

        assert len(result) == 2
        assert all(kw["keyword"].strip() for kw in result)


class TestSemanticDeduplication:
    """Tests for AI semantic deduplication"""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked Gemini"""
        with patch("openkeywords.generator.genai") as mock_genai:
            mock_genai.configure = MagicMock()
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            gen = KeywordGenerator(gemini_api_key="fake-key")
            return gen

    @pytest.mark.asyncio
    async def test_semantic_dedup_removes_near_duplicates(self, generator):
        """Test AI dedup removes near-duplicates"""
        keywords = [
            {"keyword": "sign up for project management", "score": 85},
            {"keyword": "sign up project management", "score": 75},
            {"keyword": "team collaboration tools", "score": 70},
        ]

        mock_response = MagicMock()
        mock_response.text = '{"keep": ["sign up for project management", "team collaboration tools"]}'
        generator.model.generate_content = MagicMock(return_value=mock_response)

        result = await generator._deduplicate_semantic(keywords)

        assert len(result) == 2
        assert any(kw["keyword"] == "sign up for project management" for kw in result)
        assert any(kw["keyword"] == "team collaboration tools" for kw in result)

    @pytest.mark.asyncio
    async def test_semantic_dedup_handles_api_error(self, generator):
        """Test AI dedup handles API errors gracefully"""
        keywords = [
            {"keyword": "project management", "score": 80},
            {"keyword": "team collaboration", "score": 75},
        ]

        generator.model.generate_content = MagicMock(side_effect=Exception("API Error"))

        result = await generator._deduplicate_semantic(keywords)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_semantic_dedup_single_keyword(self, generator):
        """Test AI dedup with single keyword (no dedup needed)"""
        keywords = [{"keyword": "project management", "score": 80}]

        result = await generator._deduplicate_semantic(keywords)

        assert len(result) == 1
        assert result[0]["keyword"] == "project management"
