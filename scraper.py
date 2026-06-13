import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin
import time
import os

print("=== SCRIPT STARTED ===", flush=True)

headers = {
    "User-Agent": "Mozilla/5.0"
}

pages = [
    {
        "type": "press_release",
        "url": "https://nvidianews.nvidia.com/news"
    },
    {
        "type": "earnings",
        "url": "https://investor.nvidia.com/financial-info/financial-reports/default.aspx"
    },
    {
        "type": "events",
        "url": "https://investor.nvidia.com/events-and-presentations/default.aspx"
    },
    {
        "type": "sec_filing",
        "url": "https://investor.nvidia.com/financial-info/sec-filings/default.aspx"
    }
]

records = []
visited = set()

print(f"Pages to scan: {len(pages)}", flush=True)

for page in pages:

    print(f"\nScanning: {page['type']}", flush=True)
    print(f"URL: {page['url']}", flush=True)

    try:
        response = requests.get(
            page["url"],
            headers=headers,
            timeout=20
        )

        print(
            f"Status Code: {response.status_code}",
            flush=True
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        links = soup.find_all("a", href=True)

        print(
            f"Found {len(links)} links",
            flush=True
        )

        for link in links:

            title = link.get_text(strip=True)

            if len(title) < 10:
                continue

            full_url = urljoin(
                page["url"],
                link["href"]
            )

            if full_url in visited:
                continue

            visited.add(full_url)

            try:
                print(
                    f"Opening: {full_url}",
                    flush=True
                )

                article = requests.get(
                    full_url,
                    headers=headers,
                    timeout=20
                )

                article_soup = BeautifulSoup(
                    article.text,
                    "html.parser"
                )

                content = article_soup.get_text(
                    separator=" ",
                    strip=True
                )

                records.append({
                    "source_type": page["type"],
                    "title": title,
                    "url": full_url,
                    "content": content[:5000]
                })

                print(
                    f"Saved: {title[:60]}",
                    flush=True
                )

                time.sleep(1)

            except Exception as e:
                print(
                    f"Failed to scrape article: {full_url}",
                    flush=True
                )
                print(
                    f"Error: {e}",
                    flush=True
                )

    except Exception as e:
        print(
            f"Failed page: {page['url']}",
            flush=True
        )
        print(
            f"Error: {e}",
            flush=True
        )

csv_file = "nvidia_full_dataset.csv"

with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "source_type",
            "title",
            "url",
            "content"
        ]
    )

    writer.writeheader()
    writer.writerows(records)

print("\n=== FINISHED ===", flush=True)
print(
    f"Records saved: {len(records)}",
    flush=True
)

print(
    f"CSV location: {os.path.abspath(csv_file)}",
    flush=True
)