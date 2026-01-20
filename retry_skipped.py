import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Jina AI headers
jina_headers = {
    'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}'
}

def fetch_event_content(event_url):
    """Fetch detailed content for an event using Jina AI reader"""
    jina_url = f"https://r.jina.ai/{event_url}"
    try:
        response = requests.get(jina_url, headers=jina_headers, timeout=60)
        if response.status_code == 200:
            return response.text
        else:
            print(f"  Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"  Exception: {e}")
        return None

def main():
    # Load existing data
    with open('sxodim_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    events = data['events']

    # Find skipped events
    skipped = []
    for i, event in enumerate(events):
        if event.get('markdown_content') is None and event.get('url'):
            skipped.append((i, event))

    print("=" * 60)
    print("RETRY SKIPPED EVENTS")
    print("=" * 60)
    print(f"Found {len(skipped)} skipped events")
    print("=" * 60)

    fixed = 0
    for idx, (i, event) in enumerate(skipped):
        name = event['name'][:50]
        url = event['url']
        print(f"[{idx+1}/{len(skipped)}] {name}...", end=" ", flush=True)

        content = fetch_event_content(url)
        if content:
            events[i]['markdown_content'] = content
            fixed += 1
            print("OK")
        else:
            print("FAILED")

        time.sleep(1)  # Longer delay for retries

    # Save updated data
    data['events'] = events
    data['scraped_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

    with open('sxodim_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"DONE! Fixed {fixed}/{len(skipped)} events")
    print("=" * 60)

if __name__ == '__main__':
    main()
