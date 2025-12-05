"""SE Ranking API client for gap analysis"""

import os
import logging
from typing import List, Dict, Optional
from .gap_analyzer import SEORankingAPI, AEOContentGapAnalyzer

logger = logging.getLogger(__name__)


class SEORankingAPIClient:
    """
    Wrapper around SE Ranking API for keyword gap analysis.

    Handles:
    - API initialization with key from environment
    - Competitor analysis
    - Keyword gap detection
    - AEO scoring
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize SE Ranking client"""
        self.api_key = api_key or os.getenv("SERANKING_API_KEY")

        if not self.api_key:
            raise ValueError(
                "SE Ranking API key not found. "
                "Set SERANKING_API_KEY environment variable."
            )

        self.api = SEORankingAPI(self.api_key)
        self.analyzer = AEOContentGapAnalyzer(self.api)

    def get_competitors(
        self,
        domain: str,
        source: str = "us",
        limit: int = 5
    ) -> List[str]:
        """
        Get top competitor domains.

        Args:
            domain: Target domain
            source: Region code (us, uk, de, etc.)
            limit: Maximum competitors to return

        Returns:
            List of competitor domain names
        """
        try:
            competitors_data = self.api.get_competitors(domain, source, limit)
            competitors = [c.get("domain") for c in competitors_data if c.get("domain")]
            logger.info(f"Found {len(competitors)} competitors for {domain}")
            return competitors
        except Exception as e:
            logger.error(f"Error getting competitors for {domain}: {e}")
            return []

    def analyze_content_gaps(
        self,
        domain: str,
        competitors: Optional[List[str]] = None,
        source: str = "us",
        max_competitors: int = 3,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Analyze content gaps for AEO opportunities.

        Args:
            domain: Target domain
            competitors: List of competitors (auto-detect if None)
            source: Region code
            max_competitors: Max competitors if auto-detecting
            filters: Custom AEO filters

        Returns:
            List of gap keywords with AEO scores
        """
        try:
            # Get competitors if not provided
            if not competitors:
                logger.info(f"Auto-detecting top {max_competitors} competitors for {domain}")
                competitors = self.get_competitors(domain, source, max_competitors)
                if not competitors:
                    logger.warning(f"Could not auto-detect competitors for {domain}")
                    return []

            logger.info(f"Analyzing gaps vs {len(competitors)} competitors")
            gaps = self.analyzer.analyze_content_gaps(
                domain=domain,
                competitors=competitors,
                source=source,
                max_competitors=len(competitors)
            )

            # Apply custom filters if provided
            if filters:
                gaps = self.analyzer.filter_longtail_aeo(gaps, filters)

            logger.info(f"Found {len(gaps)} gap keywords")
            return gaps

        except Exception as e:
            logger.error(f"Error analyzing gaps for {domain}: {e}")
            return []

    def extract_domain(self, url: str) -> str:
        """
        Extract domain from URL.

        Args:
            url: Full URL or domain

        Returns:
            Domain name (e.g., 'example.com')
        """
        url = url.strip()

        # Remove protocol
        if "://" in url:
            url = url.split("://")[1]

        # Remove path
        url = url.split("/")[0]

        # Remove subdomain (www., api., etc.)
        parts = url.split(".")
        if len(parts) > 2 and parts[0] in ["www", "api", "app", "mail", "ftp"]:
            url = ".".join(parts[1:])

        return url

    def test_connection(self) -> bool:
        """
        Test if API connection works.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.api.get_competitors("example.com", "us", 1)
            logger.info("SE Ranking API connection successful")
            return True
        except Exception as e:
            logger.error(f"SE Ranking API connection failed: {e}")
            return False
