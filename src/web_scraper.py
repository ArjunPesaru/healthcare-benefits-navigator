"""
web_scraper.py
--------------
Scrapes benefit information from major insurance provider websites
and saves cleaned text to data/raw/scraped/ for FAISS indexing.

Usage:
    python src/web_scraper.py
"""

import os
import time
import logging
import re
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("data/raw/scraped")
REQUEST_DELAY = 2   # seconds between requests (be polite to servers)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Insurance Provider URLs ───────────────────────────────────────────────────
# Each entry: (provider_name, category, url)
SCRAPE_TARGETS = [
    # ── UnitedHealth ──────────────────────────────────────────────────────────
    ("unitedhealthcare", "benefit_summaries",
     "https://www.uhc.com/understanding-your-benefits"),
    ("unitedhealthcare", "wellness_programs",
     "https://www.uhc.com/health-and-wellness"),
    ("unitedhealthcare", "coverage_details",
     "https://www.uhc.com/employer/benefits"),
    ("unitedhealthcare", "prescription_coverage",
     "https://www.uhc.com/pharmacy-benefits"),

    # ── Aetna ─────────────────────────────────────────────────────────────────
    ("aetna", "benefit_summaries",
     "https://www.aetna.com/individuals-families/using-your-aetna-benefits.html"),
    ("aetna", "wellness_programs",
     "https://www.aetna.com/individuals-families/member-rights-resources/tools-resources.html"),
    ("aetna", "coverage_details",
     "https://www.aetna.com/individuals-families/health-insurance-plans.html"),
    ("aetna", "prescription_coverage",
     "https://www.aetna.com/individuals-families/prescription-drug-coverage.html"),

    # ── BlueCross BlueShield ──────────────────────────────────────────────────
    ("bcbs", "benefit_summaries",
     "https://www.bcbs.com/the-health-of-america/health-of-america-report"),
    ("bcbs", "wellness_programs",
     "https://www.bcbs.com/bcbs-resources/health-wellness"),
    ("bcbs", "coverage_details",
     "https://www.bcbs.com/find-a-plan"),
    ("bcbs", "prescription_coverage",
     "https://www.bcbs.com/member-resources/pharmacy-benefits"),

    # ── Humana ────────────────────────────────────────────────────────────────
    ("humana", "benefit_summaries",
     "https://www.humana.com/member/benefits"),
    ("humana", "wellness_programs",
     "https://www.humana.com/health-and-well-being"),
    ("humana", "coverage_details",
     "https://www.humana.com/health-insurance"),
    ("humana", "prescription_coverage",
     "https://www.humana.com/pharmacy"),

    # ── Cigna ─────────────────────────────────────────────────────────────────
    ("cigna", "benefit_summaries",
     "https://www.cigna.com/individuals-families/member-guide/using-your-plan"),
    ("cigna", "wellness_programs",
     "https://www.cigna.com/individuals-families/health-wellness"),
    ("cigna", "coverage_details",
     "https://www.cigna.com/individuals-families/insurance-plans"),
    ("cigna", "prescription_coverage",
     "https://www.cigna.com/individuals-families/insurance-plans/prescription-drug"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.exceptions.HTTPError as e:
        log.warning(f"HTTP error {e.response.status_code} for {url}")
    except requests.exceptions.ConnectionError:
        log.warning(f"Connection error for {url}")
    except requests.exceptions.Timeout:
        log.warning(f"Timeout for {url}")
    except Exception as e:
        log.warning(f"Unexpected error for {url}: {e}")
    return None


def extract_text(soup: BeautifulSoup) -> str:
    """Extract clean, relevant text from a BeautifulSoup object."""
    # Remove noise tags
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "iframe", "noscript"]):
        tag.decompose()

    # Focus on main content areas if present
    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id=re.compile(r"(content|main|body)", re.I)) or
        soup.find(class_=re.compile(r"(content|main|body)", re.I)) or
        soup.body
    )

    raw = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

    # Clean up whitespace
    lines = [line.strip() for line in raw.splitlines()]
    lines = [l for l in lines if len(l) > 30]   # drop short/empty lines
    return "\n".join(lines)


def save_text(text: str, provider: str, category: str) -> Path:
    """Save extracted text to data/raw/scraped/<provider>/."""
    provider_dir = OUTPUT_DIR / provider
    provider_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{category}_{timestamp}.txt"
    out_path = provider_dir / filename

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Provider: {provider}\n")
        f.write(f"Category: {category}\n")
        f.write(f"Scraped: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)

    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def run_scraper():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Starting scraper — {len(SCRAPE_TARGETS)} targets")

    success, failed = 0, 0

    for provider, category, url in SCRAPE_TARGETS:
        log.info(f"Scraping [{provider}] {category} → {url}")
        soup = fetch_page(url)

        if soup is None:
            log.warning(f"  ✗ Skipped: {provider}/{category}")
            failed += 1
        else:
            text = extract_text(soup)
            if len(text) < 200:
                log.warning(f"  ✗ Too little text extracted ({len(text)} chars), skipping")
                failed += 1
            else:
                out_path = save_text(text, provider, category)
                log.info(f"  ✓ Saved {len(text):,} chars → {out_path}")
                success += 1

        time.sleep(REQUEST_DELAY)   # be polite

    log.info(f"\nDone! ✓  Success: {success} | Failed: {failed}")
    log.info(f"Scraped files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_scraper()