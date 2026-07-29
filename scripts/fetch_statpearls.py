"""Fetch StatPearls articles from NCBI Bookshelf; save raw text per condition."""

import os
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from config.condition_mapping import CONDITION_MAPPING

RAW_DIR = Path("data/knowledge_base/raw/statpearls")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": f"Thoragrid-RAG/1.0 ({os.getenv('NCBI_EMAIL', 'noemail')})"
}
DELAY_SEC = 2.0  # NCBI allows max 3 req/s without API key; 2s is safe


def _extract_sections(soup: BeautifulSoup) -> str:
    """Extract heading + paragraph text from NCBI Bookshelf article HTML."""
    # Remove noise elements
    for tag in soup(["script", "style", "nav", "figure", "table", "sup"]):
        tag.decompose()

    body = (
        soup.find("div", class_="article-body")
        or soup.find("div", id="maincontent")
        or soup.find("main")
    )
    if not body:
        return ""

    lines = []
    for el in body.find_all(["h2", "h3", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if not text or len(text) < 20:
            continue
        if el.name in ("h2", "h3"):
            lines.append(f"\n## {text}\n")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
            
    return "\n".join(lines)


def fetch_article(nbk_id: str) -> str:
    """Fetch one StatPearls article by NBK ID and return extracted text."""
    url = f"https://www.ncbi.nlm.nih.gov/books/{nbk_id}/"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return _extract_sections(soup)


def main():
    """Fetch all api-method conditions; skip manual and already-fetched."""
    for condition, meta in CONDITION_MAPPING.items():
        if meta["fetch"] != "api":
            print(f"skip  {condition} (manual)")
            continue

        out_path = RAW_DIR / f"{condition}.txt"
        if out_path.exists():
            print(f"exist {condition}")
            continue

        nbk_id = meta["nbk_id"]
        print(f"fetch {condition} ({nbk_id}) ...", end=" ", flush=True)
        try:
            text = fetch_article(nbk_id)
            if len(text) < 500:
                print(f"WARNING: suspiciously short ({len(text)} chars)")
            else:
                out_path.write_text(text, encoding="utf-8")
                print(f"ok ({len(text):,} chars)")
        except Exception as exc:
            print(f"ERROR: {exc}")

        time.sleep(DELAY_SEC)


if __name__ == "__main__":
    main()