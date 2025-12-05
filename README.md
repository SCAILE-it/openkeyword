# 🔑 OpenKeywords

**AI-powered SEO keyword generation using Google Gemini + SE Ranking + Deep Research**

Generate high-quality, clustered SEO keywords for any business in any language.

## ✨ Features

- **🔍 Deep Research** - Find hyper-niche keywords from Reddit, Quora, forums using Google Search grounding
- **AI Keyword Generation** - Google Gemini generates diverse, relevant keywords
- **Intent Classification** - Automatic classification (question, commercial, transactional, comparison, informational)
- **Company-Fit Scoring** - AI scores each keyword's relevance (0-100)
- **Semantic Clustering** - Groups keywords into topic clusters
- **Two-Stage Deduplication** - Fast token-based + AI semantic deduplication
- **SE Ranking Gap Analysis** - Find competitor keyword gaps (optional)
- **Any Language** - Dynamic language support, no hardcoded lists
- **AEO Optimized** - Prioritizes question keywords for Answer Engine Optimization

## 🚀 Quick Start

### Installation

```bash
pip install openkeywords
```

Or install from source:

```bash
git clone https://github.com/SCAILE-it/openkeyword.git
cd openkeyword
pip install -e .
```

### Set API Keys

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export SERANKING_API_KEY="your-seranking-key"  # Optional - for gap analysis
```

### CLI Usage

```bash
# Basic generation
openkeywords generate \
  --company "Acme Software" \
  --industry "B2B SaaS" \
  --services "project management,team collaboration" \
  --count 50

# 🔍 With Deep Research (Reddit, Quora, forums)
openkeywords generate \
  --company "Acme Software" \
  --industry "B2B SaaS" \
  --services "project management" \
  --count 50 \
  --with-research

# With SE Ranking gap analysis (requires URL + API key)
openkeywords generate \
  --company "Acme Software" \
  --url "https://acme.com" \
  --count 50 \
  --with-gaps

# Full power: Research + Gap Analysis + AI
openkeywords generate \
  --company "Acme Software" \
  --url "https://acme.com" \
  --industry "B2B SaaS" \
  --with-research \
  --with-gaps \
  --count 100

# Specify language and region
openkeywords generate \
  --company "SCAILE Technologies" \
  --industry "AEO Marketing" \
  --language "german" \
  --region "de" \
  --count 30

# Output to file
openkeywords generate \
  --company "Acme Software" \
  --count 50 \
  --output keywords.csv

# Check configuration
openkeywords check
```

## 🔍 Deep Research

Deep Research uses **Google Search grounding** to find hyper-niche keywords from real user discussions.

**What it searches:**
- **Reddit** - Real user pain points, questions, and terminology
- **Quora + PAA** - Actual questions people ask (People Also Ask)
- **Forums & Communities** - Niche industry terms and use cases

**Why it matters:**
- Finds keywords AI alone would **never generate**
- Discovers the **exact language** your audience uses
- Uncovers **long-tail, low-competition** opportunities
- Perfect for AEO (Answer Engine Optimization)

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- [SE Ranking API](https://seranking.com/api-documentation.html)
- [Google Gemini](https://ai.google.dev/)
- [SCAILE Technologies](https://scaile.tech)
