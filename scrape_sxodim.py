import requests
import json
import time
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# API headers
api_headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:147.0) Gecko/20100101 Firefox/147.0',
    'Accept': 'application/json',
    'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Referer': 'https://sxodim.com/almaty',
    'X-Requested-With': 'XMLHttpRequest',
}

# Jina AI headers
jina_headers = {
    'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}'
}

def fetch_page(page_num):
    """Fetch a single page of events from the API"""
    url = f"https://sxodim.com/api/posts/in/almaty/tickets?page={page_num}"
    response = requests.get(url, headers=api_headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching page {page_num}: {response.status_code}")
        return None

def fetch_event_content(event_url):
    """Fetch detailed content for an event using Jina AI reader"""
    jina_url = f"https://r.jina.ai/{event_url}"
    try:
        response = requests.get(jina_url, headers=jina_headers, timeout=30)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        return None

def main():
    all_events = []
    total_events_count = 0

    # First, count total events
    print("=" * 60)
    print("SXODIM.COM SCRAPER")
    print("=" * 60)

    first_page = fetch_page(1)
    if first_page and 'meta' in first_page:
        total_events_count = first_page['meta']['total']
        total_pages = first_page['meta']['last_page']
        print(f"Total events to scrape: {total_events_count}")
        print(f"Total pages: {total_pages}")
    print("=" * 60)

    current_event = 0

    # Fetch all pages (1-7)
    for page in range(1, 8):
        print(f"\n[PAGE {page}/7] Fetching...")
        page_data = fetch_page(page)

        if page_data and 'data' in page_data:
            events = page_data['data']
            print(f"[PAGE {page}/7] Found {len(events)} events")

            for i, event in enumerate(events):
                current_event += 1
                event_name = event.get('name', 'Unknown')[:40]

                # Extract key info
                event_info = {
                    'id': event.get('id'),
                    'name': event.get('name'),
                    'slug': event.get('slug'),
                    'city': event.get('city', {}).get('name'),
                    'category': event.get('category', {}).get('name'),
                    'description': event.get('description'),
                    'content_html': event.get('content'),
                    'image': event.get('image'),
                    'type': event.get('type'),
                    'subtype': event.get('subtype'),
                    'address': event.get('address'),
                    'ticket_price': event.get('additional', {}).get('ticket_price'),
                    'event_dates': event.get('event_dates', []),
                    'url': event.get('cardData', {}).get('url'),
                    'ticket_url': event.get('cardData', {}).get('ticketUrl'),
                    'views': event.get('cardData', {}).get('views'),
                    'coordinates': event.get('coordinates'),
                }

                # Fetch detailed markdown content
                print(f"  [{current_event}/{total_events_count}] {event_name}...", end=" ", flush=True)

                if event_info['url']:
                    markdown_content = fetch_event_content(event_info['url'])
                    event_info['markdown_content'] = markdown_content
                    if markdown_content:
                        print("OK")
                    else:
                        print("SKIP")
                    time.sleep(0.3)
                else:
                    event_info['markdown_content'] = None
                    print("NO URL")

                all_events.append(event_info)

                # Save progress every 10 events
                if current_event % 10 == 0:
                    with open('sxodim_data.json', 'w', encoding='utf-8') as f:
                        json.dump({
                            'total_events': len(all_events),
                            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'events': all_events
                        }, f, ensure_ascii=False, indent=2)
                    print(f"  [SAVED] Progress saved ({current_event} events)")

        time.sleep(0.5)

    # Final save
    output_file = 'sxodim_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_events': len(all_events),
            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'events': all_events
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"DONE! Saved {len(all_events)} events to {output_file}")
    print("=" * 60)

if __name__ == '__main__':
    main()
