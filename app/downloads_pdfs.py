import os
import time
import re
import logging
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ==========================
# CONFIG
# ==========================


ARTICLE_URLS = [
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12270588/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12443935/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12312990/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11931068/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11823376/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12259682/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12469573/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11632627/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12570521/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12181874/",

    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11822619/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12000858/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12086803/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12117996/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12396805/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12389004/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12270453/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11949333/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11833648/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11810274/",

    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11987642/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11876511/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12160329/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12398448/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12141479/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12023478/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12598900/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12642075/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12455369/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12542826/",

    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11800900/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11790333/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12009735/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12100291/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12283490/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11856534/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12273842/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12105097/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11842776/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12240435/",

    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11983759/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11788900/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12345678/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12077166/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11966012/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11822917/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12270722/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12532791/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12163314/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11820484/",

    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11900422/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11966788/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11884777/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11817702/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12058444/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12421499/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12177628/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12499012/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12378101/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12128891/",

    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11865440/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11922811/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12119944/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12266112/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12034988/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12500318/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12289991/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11999020/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12009008/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11777818/",

    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12399375/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11855433/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12131122/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12077452/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12244489/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12301099/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11844010/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11817745/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12451128/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12199801/",

    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11920456/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12070271/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11899880/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11822741/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12350442/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11974800/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11899802/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12100137/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11811908/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12511225/",

    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12520114/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12600789/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12447733/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12489110/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12140255/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11899881/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11833717/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12211952/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11809920/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11888219/",
]

# Folder where PDFs will be saved
OUTPUT_DIR = "pdfs"

# Delay between requests (seconds) – be polite
REQUEST_DELAY = 2.0

# Custom headers (helps avoid being mistaken for a bot/scraper)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MediExplainPDFBot/1.0; +https://example.com)"
}

# ==========================
# HELPER FUNCTIONS
# ==========================

def slugify(value: str) -> str:
    """
    Turn a string into a safe filename slug.
    """
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_-]+", "_", value)
    value = re.sub(r"^_+|_+$", "", value)
    return value or "file"


def find_pdf_link(html: str, base_url: str) -> str | None:
    """
    Try to find a PDF link in the HTML page.
    Heuristics:
      - Any <a> tag where href ends with .pdf
      - Or href contains 'pdf'
      - Or link text contains 'PDF'
    Returns the absolute URL to the PDF or None.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) Direct .pdf in href
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            return urljoin(base_url, href)

    # 2) href contains 'pdf'
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if "pdf" in href:
            return urljoin(base_url, a["href"])

    # 3) link text says 'PDF'
    for a in soup.find_all("a", href=True):
        text = (a.get_text() or "").strip().lower()
        if "pdf" in text:
            return urljoin(base_url, a["href"])

    # Special cases for common domains (fallback patterns)
    parsed = urlparse(base_url)
    domain = parsed.netloc.lower()

    # arXiv: add '.pdf' if needed
    if "arxiv.org" in domain and "/abs/" in base_url:
        return base_url.replace("/abs/", "/pdf/") + ".pdf"

    # PMC: direct /pdf/ variant
    if "pmc.ncbi.nlm.nih.gov" in domain:
        m = re.search(r"/articles/([^/]+)/", base_url)
        if m:
            pmcid = m.group(1)
            return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"

    # If we can't find anything
    return None


def get_page(url: str) -> str | None:
    """
    Fetch HTML of a page, return text or None on failure.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type", ""):
            return resp.text
        logging.warning(f"Non-HTML or bad status for {url}: {resp.status_code}")
    except Exception as e:
        logging.error(f"Error fetching page {url}: {e}")
    return None


def download_pdf(pdf_url: str, out_path: str) -> bool:
    """
    Download a single PDF and save to out_path.
    Returns True if successful.
    """
    try:
        with requests.get(pdf_url, headers=HEADERS, timeout=30, stream=True) as r:
            if r.status_code != 200:
                logging.error(f"Failed to download {pdf_url}: HTTP {r.status_code}")
                return False

            # Simple content-type check
            ctype = r.headers.get("Content-Type", "").lower()
            if "pdf" not in ctype and not pdf_url.lower().endswith(".pdf"):
                logging.warning(f"Content-Type for {pdf_url} is not PDF-ish: {ctype}")

            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        logging.info(f"Saved PDF: {out_path}")
        return True
    except Exception as e:
        logging.error(f"Error downloading PDF {pdf_url}: {e}")
        return False


def infer_filename_from_page(html: str, url: str, index: int) -> str:
    """
    Try to build a meaningful filename using <title>.
    Fallback: hostname_index.pdf
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string if soup.title and soup.title.string else ""
    if title:
        slug = slugify(title)
    else:
        hostname = urlparse(url).netloc
        slug = slugify(f"{hostname}_{index}")
    return slug + ".pdf"


# ==========================
# MAIN SCRIPT
# ==========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    for idx, article_url in enumerate(ARTICLE_URLS, start=1):
        logging.info(f"[{idx}/{len(ARTICLE_URLS)}] Processing: {article_url}")

        html = get_page(article_url)
        if not html:
            logging.error(f"Skipping {article_url} (could not fetch HTML).")
            time.sleep(REQUEST_DELAY)
            continue

        pdf_link = find_pdf_link(html, article_url)
        if not pdf_link:
            logging.error(f"No PDF link found on page: {article_url}")
            time.sleep(REQUEST_DELAY)
            continue

        filename = infer_filename_from_page(html, article_url, idx)
        out_path = os.path.join(OUTPUT_DIR, filename)

        logging.info(f"Found PDF link: {pdf_link}")
        success = download_pdf(pdf_link, out_path)

        if not success:
            logging.error(f"Failed to download PDF for {article_url}")

        time.sleep(REQUEST_DELAY)  # Be polite between requests


if __name__ == "__main__":
    main()
