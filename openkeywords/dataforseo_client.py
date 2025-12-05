# ABOUTME: Standalone DataForSEO client for SERP analysis
# ABOUTME: Provides featured snippets, PAA, related searches for AEO scoring

import base64
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any

import httpx

logger = logging.getLogger(__name__)


# DataForSEO location codes for common countries
LOCATION_CODES = {
    "us": 2840,  # United States
    "uk": 2826,  # United Kingdom
    "gb": 2826,  # United Kingdom (alt)
    "ca": 2124,  # Canada
    "au": 2036,  # Australia
    "de": 2276,  # Germany
    "fr": 2250,  # France
    "es": 2724,  # Spain
    "it": 2380,  # Italy
    "jp": 2392,  # Japan
    "br": 2076,  # Brazil
    "in": 2356,  # India
    "mx": 2484,  # Mexico
    "nl": 2528,  # Netherlands
    "se": 2752,  # Sweden
    "pl": 2616,  # Poland
    "ch": 2756,  # Switzerland
    "at": 2040,  # Austria
    "be": 2056,  # Belgium
    "pt": 2620,  # Portugal
    "dk": 2208,  # Denmark
    "no": 2578,  # Norway
    "fi": 2246,  # Finland
    "ie": 2372,  # Ireland
    "nz": 2554,  # New Zealand
    "sg": 2702,  # Singapore
    "hk": 2344,  # Hong Kong
    "kr": 2410,  # South Korea
    "tw": 2158,  # Taiwan
    "ae": 2784,  # UAE
    "za": 2710,  # South Africa
    "ar": 2032,  # Argentina
    "cl": 2152,  # Chile
    "co": 2170,  # Colombia
}

# Language codes for DataForSEO
LANGUAGE_CODES = {
    "en": "en",
    "de": "de",
    "fr": "fr",
    "es": "es",
    "it": "it",
    "pt": "pt",
    "nl": "nl",
    "pl": "pl",
    "ru": "ru",
    "ja": "ja",
    "zh": "zh-CN",
    "ko": "ko",
    "ar": "ar",
    "hi": "hi",
    "tr": "tr",
    "vi": "vi",
    "th": "th",
    "sv": "sv",
    "da": "da",
    "no": "no",
    "fi": "fi",
    "cs": "cs",
    "hu": "hu",
    "el": "el",
    "he": "he",
    "id": "id",
    "ms": "ms",
}


@dataclass
class SearchResult:
    """A single search result from SERP."""
    position: int
    title: str
    link: str
    snippet: str
    displayed_link: str = ""


@dataclass
class SerpResponse:
    """Response from a SERP query."""
    success: bool
    query: str
    results: list[SearchResult] = field(default_factory=list)
    cost: float = 0.0
    error: Optional[str] = None
    featured_snippet: Optional[dict] = None
    people_also_ask: list[dict] = field(default_factory=list)
    related_searches: list[dict] = field(default_factory=list)
    total_results: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "query": self.query,
            "results": [
                {
                    "position": r.position,
                    "title": r.title,
                    "link": r.link,
                    "snippet": r.snippet,
                    "displayed_link": r.displayed_link,
                }
                for r in self.results
            ],
            "cost": self.cost,
            "error": self.error,
            "featured_snippet": self.featured_snippet,
            "people_also_ask": self.people_also_ask,
            "related_searches": self.related_searches,
            "total_results": self.total_results,
            "timestamp": self.timestamp,
        }


class DataForSEOClient:
    """
    Client for DataForSEO API.
    
    Provides SERP analysis including:
    - Organic search results
    - Featured snippets
    - People Also Ask questions
    - Related searches
    - Keyword volume and difficulty data
    
    Requires DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment variables.
    """
    
    SERP_URL = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
    KEYWORDS_URL = "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live"
    
    def __init__(self, login: Optional[str] = None, password: Optional[str] = None):
        self.api_login = login or os.getenv("DATAFORSEO_LOGIN")
        self.api_password = password or os.getenv("DATAFORSEO_PASSWORD")
        
        if self.is_configured():
            logger.info("DataForSEO client initialized")
        else:
            logger.warning(
                "DataForSEO not configured. "
                "Set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment variables."
            )
    
    def is_configured(self) -> bool:
        """Check if DataForSEO credentials are configured."""
        return bool(self.api_login and self.api_password)
    
    def _get_auth_header(self) -> str:
        """Get Basic auth header for DataForSEO API."""
        credentials = f"{self.api_login}:{self.api_password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        language: str = "en",
        country: str = "us",
    ) -> SerpResponse:
        """
        Perform a SERP search and get rich results.
        
        Args:
            query: Search query
            num_results: Number of results to return (max 100)
            language: Language code (e.g., "en", "de")
            country: Country code (e.g., "us", "de")
        
        Returns:
            SerpResponse with results, featured snippets, PAA, and related searches
        """
        if not self.is_configured():
            return SerpResponse(
                success=False,
                query=query,
                error="DataForSEO credentials not configured",
            )
        
        try:
            location_code = LOCATION_CODES.get(country.lower(), 2840)
            lang_code = LANGUAGE_CODES.get(language.lower(), language.lower())
            
            payload = [
                {
                    "keyword": query,
                    "location_code": location_code,
                    "language_code": lang_code,
                    "depth": min(num_results, 100),
                }
            ]
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.SERP_URL,
                    json=payload,
                    headers={
                        "Authorization": self._get_auth_header(),
                        "Content-Type": "application/json",
                    },
                )
                
                if response.status_code in (401, 403):
                    return SerpResponse(
                        success=False,
                        query=query,
                        error="DataForSEO authentication failed",
                    )
                
                response.raise_for_status()
                data = response.json()
                
                # Parse the response
                return self._parse_response(data, query)
        
        except httpx.TimeoutException:
            return SerpResponse(
                success=False,
                query=query,
                error="DataForSEO request timeout",
            )
        except Exception as e:
            logger.error(f"DataForSEO search error: {e}")
            return SerpResponse(
                success=False,
                query=query,
                error=f"DataForSEO error: {str(e)}",
            )
    
    def _parse_response(self, data: dict, query: str) -> SerpResponse:
        """Parse DataForSEO response into standardized format."""
        results = []
        featured_snippet = None
        people_also_ask = []
        related_searches = []
        
        # Get the task result
        tasks = data.get("tasks", [])
        if not tasks or tasks[0].get("status_code") != 20000:
            error = tasks[0].get("status_message") if tasks else "Unknown error"
            return SerpResponse(success=False, query=query, error=error)
        
        result_data = tasks[0].get("result", [])
        if not result_data:
            return SerpResponse(success=True, query=query, results=[])
        
        items = result_data[0].get("items", [])
        
        for item in items:
            item_type = item.get("type", "")
            
            # Organic results
            if item_type == "organic":
                results.append(SearchResult(
                    position=item.get("rank_absolute", 0),
                    title=item.get("title", ""),
                    link=item.get("url", ""),
                    snippet=item.get("description", ""),
                    displayed_link=item.get("breadcrumb", ""),
                ))
            
            # Featured snippet
            elif item_type == "featured_snippet" and not featured_snippet:
                featured_snippet = {
                    "title": item.get("title"),
                    "snippet": item.get("description"),
                    "link": item.get("url"),
                }
            
            # People Also Ask
            elif item_type == "people_also_ask":
                paa_items = item.get("items", [])
                for paa in paa_items:
                    people_also_ask.append({
                        "question": paa.get("title"),
                        "snippet": paa.get("description"),
                        "link": paa.get("url"),
                    })
            
            # Related searches
            elif item_type == "related_searches":
                rs_items = item.get("items", [])
                for rs in rs_items:
                    if isinstance(rs, str):
                        related_searches.append({"query": rs})
                    elif isinstance(rs, dict):
                        related_searches.append({"query": rs.get("title")})
        
        # Cost: $0.50 per 1,000 queries = $0.0005 per query
        cost = 0.0005
        
        return SerpResponse(
            success=True,
            query=query,
            results=results,
            cost=cost,
            featured_snippet=featured_snippet,
            people_also_ask=people_also_ask,
            related_searches=related_searches,
            total_results=len(results),
        )
    
    async def get_keyword_data(
        self,
        keywords: list[str],
        language: str = "en",
        country: str = "us",
    ) -> dict[str, dict]:
        """
        Get search volume, CPC, and competition data for keywords.
        
        Uses DataForSEO Keywords Data API.
        Cost: ~$0.075 per 1000 keywords
        
        Args:
            keywords: List of keywords to analyze (max 1000 per request)
            language: Language code (e.g., "en", "de")
            country: Country code (e.g., "us", "de")
        
        Returns:
            Dict mapping keyword -> {volume, cpc, competition, difficulty}
        """
        if not self.is_configured():
            logger.warning("DataForSEO not configured for keyword data")
            return {}
        
        if not keywords:
            return {}
        
        # Limit to 1000 keywords per request (API limit)
        keywords = keywords[:1000]
        
        try:
            credentials = f"{self.api_login}:{self.api_password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            auth_header = f"Basic {encoded_credentials}"
            
            location_code = LOCATION_CODES.get(country.lower(), 2840)
            
            # Build request payload
            payload = [
                {
                    "keywords": keywords,
                    "location_code": location_code,
                    "language_code": language,
                }
            ]
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live",
                    json=payload,
                    headers={
                        "Authorization": auth_header,
                        "Content-Type": "application/json",
                    },
                )
                
                if response.status_code in (401, 403):
                    logger.error("DataForSEO authentication failed for keyword data")
                    return {}
                
                response.raise_for_status()
                data = response.json()
                
                # Parse response
                result_map = {}
                
                if data.get("tasks"):
                    for task in data["tasks"]:
                        if task.get("status_code") == 20000 and task.get("result"):
                            for item in task["result"]:
                                keyword = item.get("keyword", "").lower()
                                if keyword:
                                    # Handle competition - can be float or None
                                    competition = item.get("competition")
                                    if competition is None or not isinstance(competition, (int, float)):
                                        competition = 0.0
                                    
                                    # Competition level is a string like "LOW", "MEDIUM", "HIGH"
                                    comp_level = item.get("competition_level", "")
                                    
                                    # Estimate difficulty from competition level string
                                    difficulty_map = {"LOW": 25, "MEDIUM": 50, "HIGH": 75}
                                    difficulty = difficulty_map.get(str(comp_level).upper(), 50)
                                    
                                    result_map[keyword] = {
                                        "volume": item.get("search_volume", 0) or 0,
                                        "cpc": item.get("cpc", 0) or 0,
                                        "competition": float(competition),
                                        "competition_level": str(comp_level),
                                        "difficulty": difficulty,
                                    }
                
                logger.info(f"Got keyword data for {len(result_map)}/{len(keywords)} keywords")
                return result_map
        
        except httpx.TimeoutException:
            logger.error("DataForSEO keyword data request timeout")
            return {}
        except Exception as e:
            logger.error(f"DataForSEO keyword data error: {e}")
            return {}
    
    async def get_keyword_difficulty(
        self,
        keywords: list[str],
        language: str = "en",
        country: str = "us",
    ) -> dict[str, int]:
        """
        Get keyword difficulty scores (0-100).
        
        Uses DataForSEO Keyword Difficulty API for more accurate scores.
        Cost: ~$0.05 per keyword
        
        Args:
            keywords: List of keywords (max 1000)
            language: Language code
            country: Country code
        
        Returns:
            Dict mapping keyword -> difficulty (0-100)
        """
        if not self.is_configured():
            return {}
        
        if not keywords:
            return {}
        
        keywords = keywords[:1000]
        
        try:
            credentials = f"{self.api_login}:{self.api_password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            auth_header = f"Basic {encoded_credentials}"
            
            location_code = LOCATION_CODES.get(country.lower(), 2840)
            
            # Build batch request - one keyword per task for difficulty
            payload = [
                {
                    "keyword": kw,
                    "location_code": location_code,
                    "language_code": language,
                }
                for kw in keywords
            ]
            
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    "https://api.dataforseo.com/v3/dataforseo_labs/google/keyword_difficulty/live",
                    json=payload,
                    headers={
                        "Authorization": auth_header,
                        "Content-Type": "application/json",
                    },
                )
                
                if response.status_code in (401, 403):
                    logger.error("DataForSEO authentication failed for keyword difficulty")
                    return {}
                
                response.raise_for_status()
                data = response.json()
                
                result_map = {}
                
                if data.get("tasks"):
                    for task in data["tasks"]:
                        if task.get("status_code") == 20000 and task.get("result"):
                            for item in task["result"]:
                                keyword = item.get("keyword", "").lower()
                                difficulty = item.get("keyword_difficulty", 50)
                                if keyword:
                                    result_map[keyword] = int(difficulty) if difficulty else 50
                
                logger.info(f"Got difficulty for {len(result_map)}/{len(keywords)} keywords")
                return result_map
        
        except Exception as e:
            logger.error(f"DataForSEO keyword difficulty error: {e}")
            return {}


# Convenience function for one-off searches
async def search_serp(
    query: str,
    country: str = "us",
    language: str = "en",
    login: Optional[str] = None,
    password: Optional[str] = None,
) -> SerpResponse:
    """
    Convenience function for single SERP search.
    
    Args:
        query: Search query
        country: Country code
        language: Language code
        login: Optional DataForSEO login (uses env var if not provided)
        password: Optional DataForSEO password (uses env var if not provided)
    
    Returns:
        SerpResponse with results and SERP features
    """
    client = DataForSEOClient(login=login, password=password)
    return await client.search(query, country=country, language=language)
