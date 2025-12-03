import os
import time
import re
import logging

import requests
from pypdf import PdfReader

# =====================================================
# CONFIG – 100 PMC OPEN-ACCESS MEDICAL ARTICLE LINKS
# =====================================================
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

OUTPUT_DIR = "pdfs"
REQUEST_DELAY = 2.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MediExplainPDFBot/1.0; +https://example.com)"
}

# =====================================================
# HELPERS
# =====================================================

def pmc_to_pdf_url(article_url: str):
    """
    Convert a PMC article URL → direct PDF URL + filename.
    Example:
      https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/
      → https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/
    """
    clean = article_url.split("?")[0].rstrip("/")
    m = re.search(r"/articles/(PMC[0-9]+)/?$", clean)
    if not m:
        logging.error(f"Could not extract PMCID from {article_url}")
        return None
    pmcid = m.group(1)
    pdf_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"
    filename = f"{pmcid}.pdf"
    return pdf_url, filename


def is_valid_pdf(path: str) -> bool:
    """Use pypdf to verify the file is a real, readable PDF."""
    try:
        reader = PdfReader(path)
        _ = len(reader.pages)  # force lazy load
        return True
    except Exception as e:
        logging.error(f"File is not a valid PDF ({path}): {e}")
        return False


def download_pdf(pdf_url: str, out_path: str) -> bool:
    """Download the PDF, check Content-Type AND validate with pypdf."""
    try:
        with requests.get(pdf_url, headers=HEADERS, timeout=30, stream=True) as r:
            if r.status_code != 200:
                logging.error(f"Failed: {pdf_url}: HTTP {r.status_code}")
                return False

            ctype = r.headers.get("Content-Type", "").lower()
            # PMC usually returns application/pdf; occasionally others may appear
            if "pdf" not in ctype and "application/octet-stream" not in ctype:
                logging.error(f"Not a PDF at {pdf_url} (Content-Type={ctype})")
                return False

            # write to disk
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

        # now validate with pypdf
        if not is_valid_pdf(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
            return False

        logging.info(f"Saved valid PDF: {out_path}")
        return True

    except Exception as e:
        logging.error(f"Download error for {pdf_url}: {e}")
        # cleanup partial file
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return False


# =====================================================
# MAIN SCRIPT
# =====================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    success_count = 0
    fail_count = 0
    skipped_existing = 0
    failed_urls = []

    for idx, article_url in enumerate(ARTICLE_URLS, start=1):
        logging.info(f"[{idx}/{len(ARTICLE_URLS)}] {article_url}")

        result = pmc_to_pdf_url(article_url)
        if not result:
            fail_count += 1
            failed_urls.append(article_url)
            continue

        pdf_url, filename = result
        out_path = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(out_path):
            logging.info(f"Already exists, skipping: {out_path}")
            skipped_existing += 1
        else:
            logging.info(f"Downloading: {pdf_url}")
            ok = download_pdf(pdf_url, out_path)

            if ok:
                success_count += 1
            else:
                fail_count += 1
                failed_urls.append(article_url)

            time.sleep(REQUEST_DELAY)

    logging.info(
        f"Done. Valid PDFs: {success_count}, "
        f"failed: {fail_count}, skipped existing: {skipped_existing}"
    )

    if failed_urls:
        logging.info("Failed URLs:")
        for u in failed_urls:
            logging.info(f"  {u}")


if __name__ == "__main__":
    main()
