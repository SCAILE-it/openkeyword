#!/usr/bin/env python3
"""
AEO Content Gap Analyzer
Finds long-tail, question-based content gaps optimized for Answer Engine Optimization (AEO)
Targets: ChatGPT, Google AI Overviews, Perplexity, Bing Copilot
"""

import os
import requests
import json
import csv
from typing import List, Dict, Optional
from datetime import datetime
import argparse
import sys

BASE_URL = "https://api.seranking.com/v1"

AEO_FILTERS = {
    "min_volume": 100,
    "max_volume": 5000,
    "max_difficulty": 35,
    "max_competition": 0.3,
    "min_words": 3,
}

AEO_INTENT_PATTERNS = {
    "question": {
        "keywords": ["how", "what", "why", "when", "where", "who", "can", "should", "does", "is", "are", "will", "would"],
        "multiplier": 1.5,
        "description": "Question-based queries - Perfect for featured snippets & AI answers"
    },
    "commercial": {
        "keywords": ["best", "top", "vs", "versus", "review", "compare", "pricing", "cost", "alternative"],
        "multiplier": 1.3,
        "description": "Commercial intent - Good for AI recommendations"
    },
    "informational": {
        "keywords": ["guide", "tutorial", "tips", "examples", "template", "checklist", "definition", "meaning"],
        "multiplier": 1.4,
        "description": "Informational queries - Excellent for AI citations"
    },
    "list": {
        "keywords": ["list", "ways to", "steps to", "types of", "kinds of", "ideas for"],
        "multiplier": 1.4,
        "description": "List-based queries - Perfect for structured answers"
    },
    "local": {
        "keywords": ["near me", "in", "local", "nearby", "around"],
        "multiplier": 1.1,
        "description": "Local queries"
    },
}

AEO_SERP_FEATURES = [
    "people_also_ask",
    "featured_snippet",
    "sge",
    "knowledge_panel",
    "faq"
]


class SEORankingAPI:
    """SE Ranking API client"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json"
        }
        self.base_url = BASE_URL

    def get_competitors(self, domain: str, source: str = "us", limit: int = 5) -> List[Dict]:
        """Get top competitors for a domain"""
        url = f"{self.base_url}/domain/competitors"
        params = {
            "source": source,
            "domain": domain,
            "limit": limit
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching competitors: {e}")
            return []

    def get_keyword_comparison(self, domain: str, compare_domain: str,
                               source: str = "us", limit: int = 1000) -> List[Dict]:
        """Get keywords that competitor ranks for but target domain doesn't"""
        url = f"{self.base_url}/domain/keywords/comparison"
        params = {
            "source": source,
            "domain": domain,
            "compare": compare_domain,
            "type": "organic",
            "diff": 1,
            "limit": limit,
            "order_field": "difficulty",
            "order_type": "asc"
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching keyword comparison: {e}")
            return []


class AEOContentGapAnalyzer:
    """Analyzes content gaps with AEO optimization"""

    def __init__(self, api: SEORankingAPI):
        self.api = api

    def filter_longtail_aeo(self, keywords: List[Dict],
                            filters: Optional[Dict] = None) -> List[Dict]:
        """Filter keywords for long-tail AEO opportunities"""
        if filters is None:
            filters = AEO_FILTERS

        longtail = []

        for kw in keywords:
            volume = kw.get("volume", 0)
            difficulty = kw.get("difficulty", 100)
            competition = kw.get("competition", 1)
            keyword_text = kw.get("keyword", "")
            word_count = len(keyword_text.split())

            if (filters["min_volume"] <= volume <= filters["max_volume"] and
                difficulty <= filters["max_difficulty"] and
                competition <= filters["max_competition"] and
                word_count >= filters["min_words"]):

                kw["word_count"] = word_count
                longtail.append(kw)

        return longtail

    def categorize_by_intent(self, keyword: Dict) -> Dict:
        """Categorize keyword by AEO intent and add multiplier"""
        keyword_lower = keyword["keyword"].lower()
        matched_intents = []
        max_multiplier = 1.0
        primary_intent = "other"

        for intent, config in AEO_INTENT_PATTERNS.items():
            if any(pattern in keyword_lower for pattern in config["keywords"]):
                matched_intents.append(intent)
                if config["multiplier"] > max_multiplier:
                    max_multiplier = config["multiplier"]
                    primary_intent = intent

        keyword["intent"] = primary_intent
        keyword["intent_multiplier"] = max_multiplier
        keyword["matched_intents"] = matched_intents

        return keyword

    def check_aeo_serp_features(self, keyword: Dict) -> Dict:
        """Check if keyword has AEO-friendly SERP features"""
        serp_features = keyword.get("serp_features", [])

        aeo_features = [f for f in serp_features if f in AEO_SERP_FEATURES]
        keyword["aeo_serp_features"] = aeo_features
        keyword["has_aeo_features"] = len(aeo_features) > 0

        if keyword["has_aeo_features"]:
            keyword["aeo_feature_boost"] = 1.3
        else:
            keyword["aeo_feature_boost"] = 1.0

        return keyword

    def calculate_aeo_score(self, keyword: Dict) -> float:
        """
        Calculate AEO opportunity score

        AEO Score = (Volume x Intent Multiplier x SERP Feature Boost) / (Difficulty + 1)
        """
        volume = keyword.get("volume", 0)
        difficulty = keyword.get("difficulty", 100)
        intent_multiplier = keyword.get("intent_multiplier", 1.0)
        aeo_feature_boost = keyword.get("aeo_feature_boost", 1.0)

        aeo_score = (volume * intent_multiplier * aeo_feature_boost) / (difficulty + 1)

        keyword["aeo_score"] = round(aeo_score, 2)
        return aeo_score

    def analyze_content_gaps(self, domain: str, competitors: Optional[List[str]] = None,
                            source: str = "us", max_competitors: int = 3) -> List[Dict]:
        """
        Analyze content gaps for AEO optimization
        """
        print(f"\nAnalyzing AEO Content Gaps for: {domain}")

        if not competitors:
            print(f"\nFinding top {max_competitors} competitors...")
            competitor_data = self.api.get_competitors(domain, source, max_competitors)
            competitors = [c["domain"] for c in competitor_data[:max_competitors]]
            print(f"Found competitors: {', '.join(competitors)}")

        all_gaps = []

        for i, competitor in enumerate(competitors, 1):
            print(f"\n[{i}/{len(competitors)}] Comparing with {competitor}...")

            gaps = self.api.get_keyword_comparison(domain, competitor, source)

            if gaps:
                print(f"  Found {len(gaps)} total keyword gaps")

                longtail = self.filter_longtail_aeo(gaps)
                print(f"  Filtered to {len(longtail)} long-tail AEO opportunities")

                for kw in longtail:
                    kw["competitor"] = competitor
                    self.categorize_by_intent(kw)
                    self.check_aeo_serp_features(kw)
                    self.calculate_aeo_score(kw)

                all_gaps.extend(longtail)

        all_gaps.sort(key=lambda x: x["aeo_score"], reverse=True)

        print(f"\nTotal AEO opportunities found: {len(all_gaps)}")

        return all_gaps

    def generate_summary_stats(self, gaps: List[Dict]) -> Dict:
        """Generate summary statistics"""
        if not gaps:
            return {}

        intent_counts = {}
        for gap in gaps:
            intent = gap.get("intent", "other")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        with_aeo_features = sum(1 for g in gaps if g.get("has_aeo_features", False))
        question_kw = [g for g in gaps if g.get("intent") == "question"]

        return {
            "total_opportunities": len(gaps),
            "intent_breakdown": intent_counts,
            "with_aeo_serp_features": with_aeo_features,
            "question_keywords": len(question_kw),
            "avg_aeo_score": round(sum(g["aeo_score"] for g in gaps) / len(gaps), 2),
            "avg_volume": round(sum(g["volume"] for g in gaps) / len(gaps)),
            "avg_difficulty": round(sum(g["difficulty"] for g in gaps) / len(gaps), 1),
        }

    def export_to_csv(self, gaps: List[Dict], filename: str):
        """Export results to CSV"""
        if not gaps:
            print("No gaps to export")
            return

        fieldnames = [
            "keyword", "volume", "difficulty", "cpc", "competition",
            "aeo_score", "intent", "word_count", "has_aeo_features",
            "aeo_serp_features", "competitor", "url", "position"
        ]

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(gaps)

        print(f"\nExported {len(gaps)} opportunities to: {filename}")

    def export_to_json(self, gaps: List[Dict], filename: str):
        """Export results to JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(gaps, f, indent=2, ensure_ascii=False)

        print(f"\nExported {len(gaps)} opportunities to: {filename}")

    def print_top_opportunities(self, gaps: List[Dict], top_n: int = 20):
        """Print top AEO opportunities"""
        if not gaps:
            print("\nNo opportunities found")
            return

        print(f"\nTOP {min(top_n, len(gaps))} AEO OPPORTUNITIES")

        for i, kw in enumerate(gaps[:top_n], 1):
            print(f"\n{i}. {kw['keyword']}")
            print(f"   Volume: {kw['volume']:,}/mo | Difficulty: {kw['difficulty']} | AEO Score: {kw['aeo_score']}")
            print(f"   Intent: {kw['intent'].upper()} | Words: {kw['word_count']} | Competitor: {kw['competitor']}")

            if kw.get('has_aeo_features'):
                print(f"   SERP Features: {', '.join(kw['aeo_serp_features'])}")

            if kw.get('cpc', 0) > 0:
                print(f"   CPC: ${kw['cpc']:.2f} | Competition: {kw['competition']:.2f}")
