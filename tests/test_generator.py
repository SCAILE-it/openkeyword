"""
Tests for KeywordGenerator
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
import json

from openkeywords import KeywordGenerator, CompanyInfo, GenerationConfig


class TestGeneratorInit:
    """Tests for KeywordGenerator initialization"""

    def test_init_with_api_key(self):
        """Test initialization with API key"""
        with patch("openkeywords.generator.genai") as mock_genai:
            mock_genai.configure = MagicMock()
            mock_genai.GenerativeModel = MagicMock()

            gen = KeywordGenerator(gemini_api_key="test-key")

            mock_genai.configure.assert_called_once_with(api_key="test-key")
            assert gen.api_key == "test-key"

    def test_init_without_api_key_raises(self):
        """Test initialization fails without API key"""
        with patch.dict("os.environ", {}, clear=True):
            with patch("openkeywords.generator.genai"):
                with pytest.raises(ValueError, match="Gemini API key required"):
                    KeywordGenerator()

    def test_init_with_env_var(self):
        """Test initialization with env var"""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}):
            with patch("openkeywords.generator.genai") as mock_genai:
                mock_genai.configure = MagicMock()
                mock_genai.GenerativeModel = MagicMock()

                gen = KeywordGenerator()

                assert gen.api_key == "env-key"

    def test_init_with_seranking(self):
        """Test initialization with SE Ranking key"""
        with patch("openkeywords.generator.genai") as mock_genai:
            mock_genai.configure = MagicMock()
            mock_genai.GenerativeModel = MagicMock()

            with patch.dict("sys.modules", {"openkeywords.seranking_client": MagicMock()}):
                with patch("openkeywords.seranking_client.SEORankingAPIClient") as mock_seranking:
                    mock_seranking.return_value = MagicMock()
                    
                    gen = KeywordGenerator(
                        gemini_api_key="test-key",
                        seranking_api_key="seranking-key"
                    )

                    assert gen.seranking_api_key == "seranking-key"


class TestGeneratorGenerate:
    """Tests for keyword generation"""

    @pytest.fixture
    def mock_generator(self):
        """Create generator with all mocks"""
        with patch("openkeywords.generator.genai") as mock_genai:
            mock_genai.configure = MagicMock()
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            mock_genai.GenerationConfig = MagicMock()

            gen = KeywordGenerator(gemini_api_key="fake-key")
            gen.model = mock_model
            return gen

    @pytest.mark.asyncio
    async def test_generate_basic(self, mock_generator, sample_company):
        """Test basic keyword generation"""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "keywords": [
                {"keyword": "project management software", "intent": "commercial", "is_question": False},
                {"keyword": "how to manage projects", "intent": "question", "is_question": True},
            ]
        })
        mock_generator.model.generate_content = MagicMock(return_value=mock_response)

        config = GenerationConfig(target_count=5, min_score=0)
        result = await mock_generator.generate(sample_company, config)

        assert result is not None
        assert result.processing_time_seconds > 0

    @pytest.mark.asyncio
    async def test_generate_empty_result(self, mock_generator, sample_company):
        """Test handling of empty generation result"""
        mock_response = MagicMock()
        mock_response.text = '{"keywords": []}'
        mock_generator.model.generate_content = MagicMock(return_value=mock_response)

        config = GenerationConfig(target_count=5)
        result = await mock_generator.generate(sample_company, config)

        assert result.keywords == []
        assert result.statistics.total == 0


class TestGeneratorScoring:
    """Tests for keyword scoring"""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked Gemini"""
        with patch("openkeywords.generator.genai") as mock_genai:
            mock_genai.configure = MagicMock()
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            mock_genai.GenerationConfig = MagicMock()

            gen = KeywordGenerator(gemini_api_key="fake-key")
            return gen

    @pytest.mark.asyncio
    async def test_score_keywords(self, generator, sample_company, sample_keywords):
        """Test keyword scoring"""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "scores": [
                {"keyword": "best project management software", "score": 85},
                {"keyword": "how to manage projects remotely", "score": 75},
                {"keyword": "project management tools pricing", "score": 80},
                {"keyword": "sign up for project management", "score": 70},
                {"keyword": "team collaboration software", "score": 72},
            ]
        })
        generator.model.generate_content = MagicMock(return_value=mock_response)

        keywords = [{"keyword": kw["keyword"], "intent": kw["intent"]} for kw in sample_keywords]

        result = await generator._score_keywords(keywords, sample_company)

        assert len(result) == 5
        assert all("score" in kw for kw in result)
        assert result[0]["score"] >= result[-1]["score"]

    @pytest.mark.asyncio
    async def test_score_preserves_gap_analysis(self, generator, sample_company):
        """Test that gap_analysis keywords keep their scores"""
        keywords = [
            {"keyword": "ai generated keyword", "intent": "commercial", "source": "ai_generated"},
            {"keyword": "gap analysis keyword", "intent": "commercial", "score": 90, "source": "gap_analysis"},
        ]

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "scores": [{"keyword": "ai generated keyword", "score": 75}]
        })
        generator.model.generate_content = MagicMock(return_value=mock_response)

        result = await generator._score_keywords(keywords, sample_company)

        gap_kw = next(kw for kw in result if kw["source"] == "gap_analysis")
        assert gap_kw["score"] == 90


class TestGeneratorClustering:
    """Tests for keyword clustering"""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked Gemini"""
        with patch("openkeywords.generator.genai") as mock_genai:
            mock_genai.configure = MagicMock()
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            mock_genai.GenerationConfig = MagicMock()

            gen = KeywordGenerator(gemini_api_key="fake-key")
            return gen

    @pytest.mark.asyncio
    async def test_cluster_keywords(self, generator, sample_company, sample_keywords):
        """Test keyword clustering"""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "clusters": [
                {
                    "cluster_name": "Product Features",
                    "keywords": ["best project management software", "project management tools pricing"]
                },
                {
                    "cluster_name": "How-To Guides",
                    "keywords": ["how to manage projects remotely"]
                },
                {
                    "cluster_name": "Getting Started",
                    "keywords": ["sign up for project management", "team collaboration software"]
                },
            ]
        })
        generator.model.generate_content = MagicMock(return_value=mock_response)

        result = await generator._cluster_keywords(sample_keywords, sample_company, 3)

        assert len(result) == 5
        assert all("cluster_name" in kw for kw in result)

        product_kw = next(kw for kw in result if kw["keyword"] == "best project management software")
        assert product_kw["cluster_name"] == "Product Features"

    @pytest.mark.asyncio
    async def test_cluster_handles_error(self, generator, sample_company, sample_keywords):
        """Test clustering handles errors gracefully"""
        generator.model.generate_content = MagicMock(side_effect=Exception("API Error"))

        result = await generator._cluster_keywords(sample_keywords, sample_company, 3)

        assert len(result) == 5
        assert all(kw.get("cluster_name") == "General" for kw in result)


class TestGeneratorStatistics:
    """Tests for statistics calculation"""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked Gemini"""
        with patch("openkeywords.generator.genai") as mock_genai:
            mock_genai.configure = MagicMock()
            mock_genai.GenerativeModel = MagicMock()
            gen = KeywordGenerator(gemini_api_key="fake-key")
            return gen

    def test_calculate_statistics(self, generator):
        """Test statistics calculation"""
        from openkeywords import Keyword

        keywords = [
            Keyword(keyword="short one", intent="commercial", score=80),
            Keyword(keyword="medium length keyword here", intent="question", score=70),
            Keyword(keyword="this is a longer keyword phrase", intent="informational", score=60),
            Keyword(keyword="another commercial keyword", intent="commercial", score=75),
        ]

        stats = generator._calculate_statistics(keywords, duplicate_count=3)

        assert stats.total == 4
        assert stats.avg_score == 71.25
        assert stats.duplicate_count == 3
        assert stats.intent_breakdown["commercial"] == 2
        assert stats.intent_breakdown["question"] == 1
        assert stats.intent_breakdown["informational"] == 1

    def test_calculate_statistics_empty(self, generator):
        """Test statistics with empty keywords"""
        stats = generator._calculate_statistics([], duplicate_count=0)

        assert stats.total == 0
        assert stats.avg_score == 0.0
