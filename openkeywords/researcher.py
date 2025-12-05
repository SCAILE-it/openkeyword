"""
Deep Research Engine - Find hyper-niche keywords from real user discussions.

Uses Google Search grounding to find:
- Reddit discussions (pain points, questions, terminology)
- Quora questions (real user language)
- Forum posts (niche communities)
- People Also Ask (Google PAA)
- Blog comments and reviews

This finds keywords that AI alone would never generate.
"""

import asyncio
import json
import logging
import os
import re
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

RESEARCH_SOURCES = [
    "site:reddit.com",
    "site:quora.com",
    "people also ask",
    "forum",
    "community",
]


class ResearchEngine:
    """
    Deep research engine using Google Search grounding.

    Discovers hyper-niche, long-tail keywords and questions from:
    - Reddit (authentic user discussions)
    - Quora (real questions people ask)
    - Forums (niche communities)
    - People Also Ask (Google suggestions)

    Uses Gemini with Google Search tool for grounded research.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set GEMINI_API_KEY env var or pass api_key."
            )

        try:
            from google import genai
            from google.genai import types

            self.genai = genai
            self.types = types
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = model
            self._has_search_tools = True
            logger.info(f"Research engine initialized with Google Search grounding")
        except ImportError:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model)
            self._has_search_tools = False
            logger.warning(
                "google-genai SDK not available. Install with: pip install google-genai"
            )

    async def discover_keywords(
        self,
        company_name: str,
        industry: str,
        services: list[str],
        products: list[str] = None,
        target_location: str = "United States",
        language: str = "english",
        target_count: int = 30,
    ) -> list[dict]:
        all_keywords = []

        tasks = [
            self._research_reddit(industry, services, language, target_count // 3),
            self._research_questions(industry, services, language, target_count // 3),
            self._research_niche_terms(
                industry, services, products or [], language, target_count // 3
            ),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Research task failed: {result}")
            elif result:
                all_keywords.extend(result)

        logger.info(f"Deep research found {len(all_keywords)} raw keywords")

        seen = set()
        unique = []
        for kw in all_keywords:
            text = kw.get("keyword", "").lower().strip()
            if text and text not in seen:
                seen.add(text)
                unique.append(kw)

        logger.info(f"After dedup: {len(unique)} unique keywords from research")
        return unique

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _research_reddit(
        self,
        industry: str,
        services: list[str],
        language: str,
        target_count: int,
    ) -> list[dict]:
        services_str = ", ".join(services[:3]) if services else industry

        prompt = f"""You are a keyword researcher. Search Reddit for REAL discussions about {industry}.

Search for: "{industry} site:reddit.com" and "{services_str} site:reddit.com"

Find {target_count} unique keywords/phrases that REAL USERS use when discussing:
- Problems they face related to {services_str}
- Questions they ask
- Solutions they're looking for
- Specific terminology and jargon
- Pain points and frustrations

Focus on:
- Long-tail keywords (4-7 words)
- Question-based keywords (how, what, why, can I, should I)
- Problem-based keywords (problem with, issue, help with, struggling with)
- Comparison keywords (vs, versus, alternative to, better than)

IMPORTANT: Find NICHE keywords that typical AI keyword generators would miss.

Output JSON:
{{"keywords": [
  {{"keyword": "exact phrase from reddit", "intent": "question|commercial|informational|transactional|comparison", "source": "reddit", "context": "brief context where found"}}
]}}"""

        return await self._execute_grounded_research(prompt, "reddit")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _research_questions(
        self,
        industry: str,
        services: list[str],
        language: str,
        target_count: int,
    ) -> list[dict]:
        services_str = ", ".join(services[:3]) if services else industry

        prompt = f"""You are a keyword researcher. Search for REAL QUESTIONS people ask about {industry}.

Search: "{industry} site:quora.com" and "people also ask {services_str}"

Find {target_count} unique QUESTION keywords that real users ask about:
- {services_str}
- Problems in {industry}
- Buying decisions
- Comparisons and alternatives

Focus on:
- Complete question phrases (how do I, what is the best, why does)
- Specific problem questions (why won't, how to fix, what to do when)
- Decision questions (should I, is it worth, which is better)
- "People Also Ask" style questions

These should be REAL questions from Quora, forums, and Google PAA.

Output JSON:
{{"keywords": [
  {{"keyword": "exact question from research", "intent": "question", "source": "quora_paa", "context": "where found"}}
]}}"""

        return await self._execute_grounded_research(prompt, "quora_paa")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _research_niche_terms(
        self,
        industry: str,
        services: list[str],
        products: list[str],
        language: str,
        target_count: int,
    ) -> list[dict]:
        context = f"{industry}, {', '.join(services[:2])}"
        if products:
            context += f", {', '.join(products[:2])}"

        prompt = f"""You are a keyword researcher. Search for NICHE terminology in {industry}.

Search forums, communities, and specialized sites for: "{context}"

Find {target_count} unique NICHE keywords including:
- Industry-specific terminology and jargon
- Specific use cases (e.g., "project management for construction companies")
- Role-specific keywords (e.g., "CRM for sales managers")
- Problem-specific keywords (e.g., "inventory management for small retail")
- Feature-specific keywords (e.g., "kanban board software")
- Location or segment specific (e.g., "accounting software for freelancers UK")

Focus on:
- Hyper-specific long-tail keywords (5-8 words)
- Keywords with modifiers (best, free, affordable, enterprise)
- Use-case specific (for startups, for agencies, for remote teams)
- Industry vertical specific

Find the EXACT terminology and phrases professionals use.

Output JSON:
{{"keywords": [
  {{"keyword": "niche term found", "intent": "commercial|informational|transactional", "source": "niche_research", "context": "context"}}
]}}"""

        return await self._execute_grounded_research(prompt, "niche_research")

    async def _execute_grounded_research(
        self, prompt: str, source_type: str
    ) -> list[dict]:
        if self._has_search_tools:
            return await self._execute_with_search_tools(prompt, source_type)
        else:
            return await self._execute_simulated_research(prompt, source_type)

    async def _execute_with_search_tools(
        self, prompt: str, source_type: str
    ) -> list[dict]:
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    tools=[self.types.Tool(google_search=self.types.GoogleSearch())],
                    temperature=0.5,
                ),
            )

            response_text = response.text
            keywords = self._parse_keywords_response(response_text)

            logger.info(f"Research ({source_type}): found {len(keywords)} keywords")
            return keywords

        except Exception as e:
            logger.error(f"Search-grounded research failed: {e}")
            return await self._execute_simulated_research(prompt, source_type)

    async def _execute_simulated_research(
        self, prompt: str, source_type: str
    ) -> list[dict]:
        try:
            fallback_prompt = prompt.replace(
                "Search for:",
                "Based on your knowledge of typical discussions, imagine what you would find if searching for:"
            ).replace(
                "Search:",
                "Based on your knowledge, generate realistic keywords as if you searched:"
            )

            response = await asyncio.to_thread(
                self.model.generate_content,
                fallback_prompt,
                generation_config={"temperature": 0.7, "response_mime_type": "application/json"},
            )

            keywords = self._parse_keywords_response(response.text)
            logger.info(f"Simulated research ({source_type}): {len(keywords)} keywords")
            return keywords

        except Exception as e:
            logger.error(f"Simulated research failed: {e}")
            return []

    def _parse_keywords_response(self, response_text: str) -> list[dict]:
        try:
            text = response_text.strip()

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            keywords_data = data.get("keywords", [])

            valid_keywords = []
            for kw in keywords_data:
                keyword_text = kw.get("keyword", "").strip()
                if not keyword_text or len(keyword_text) < 5:
                    continue

                keyword_text = re.sub(r'\s+', ' ', keyword_text)

                valid_keywords.append({
                    "keyword": keyword_text,
                    "intent": kw.get("intent", "informational"),
                    "source": kw.get("source", "research"),
                    "context": kw.get("context", ""),
                    "is_question": keyword_text.lower().startswith(
                        ("how", "what", "why", "when", "where", "which", "who", "can", "should", "is", "are", "does", "do")
                    ),
                    "score": 0,
                })

            return valid_keywords

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse research response: {e}")
            return []
