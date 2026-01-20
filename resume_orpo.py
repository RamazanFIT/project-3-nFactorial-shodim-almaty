import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def generate_preference_pair(event):
    """Generate chosen (friendly) and rejected (formal) response pairs"""

    event_context = f"""
Название: {event.get('name', 'Не указано')}
Категория: {event.get('category', 'Не указано')}
Адрес: {event.get('address', 'Не указано')}
Цена: {event.get('ticket_price', 'Не указано')}
Описание: {event.get('description', 'Не указано')}
URL: {event.get('url', '')}

Содержимое:
{event.get('markdown_content', event.get('content_html', ''))[:3000]}
"""

    prompt = f"""На основе информации о мероприятии создай 5 пар вопрос-ответ для обучения модели.

Информация о мероприятии:
{event_context}

Для КАЖДОГО вопроса создай ДВА варианта ответа:

1. **chosen** (предпочтительный) — дружелюбный, живой, человечный ответ:
   - Используй разговорный стиль, как будто общаешься с другом
   - Добавляй эмоции: "Очень советую!", "Это будет огонь!", "Классное место!"
   - Используй личные рекомендации: "Я бы точно сходил", "Мне кажется, тебе понравится"
   - Можно добавить юмор или интересные детали
   - Используй сокращения и неформальную речь где уместно

2. **rejected** (нежелательный) — сухой, формальный, роботизированный ответ:
   - Официальный канцелярский стиль
   - Без эмоций и личного отношения
   - Скучное перечисление фактов
   - Фразы типа "Данное мероприятие состоится...", "Информируем вас о том, что..."

Формат JSON:
[
  {{
    "prompt": "вопрос пользователя",
    "chosen": "дружелюбный живой ответ",
    "rejected": "сухой формальный ответ"
  }}
]

Отвечай ТОЛЬКО JSON массивом."""

    try:
        response = client.chat.completions.create(
            model="google/gemini-3-flash-preview",
            messages=[{"role": "user", "content": prompt}],
            extra_body={
                "reasoning": {"enabled": True},
                "provider": {"allow_fallbacks": False, "only": ["google-ai-studio"]}
            }
        )

        content = response.choices[0].message.content.strip()

        # Clean markdown
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        pairs = json.loads(content)
        return pairs

    except json.JSONDecodeError as e:
        print(f"  JSON error: {e}")
        return []
    except Exception as e:
        print(f"  API error: {e}")
        return []


def main():
    # Load existing ORPO dataset
    with open('orpo_dataset.json', 'r', encoding='utf-8') as f:
        orpo_dataset = json.load(f)

    processed_ids = set(item.get('event_id') for item in orpo_dataset)

    # Load all events
    with open('sxodim_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    events = data['events']
    missing_events = [e for e in events if e.get('id') not in processed_ids]

    print("=" * 60)
    print("ORPO RESUME - Processing missing events")
    print("=" * 60)
    print(f"Already processed: {len(processed_ids)} events")
    print(f"Missing: {len(missing_events)} events")
    print("=" * 60)

    system_prompt = "Ты — дружелюбный помощник по мероприятиям в Алматы. Помогаешь пользователям найти интересные события, концерты, спектакли, выставки и развлечения в городе. Общайся живо и по-дружески, как будто советуешь другу куда сходить."

    for i, event in enumerate(missing_events):
        name = event.get('name', 'Unknown')[:45]
        print(f"[{i+1}/{len(missing_events)}] {name}...", end=" ", flush=True)

        pairs = generate_preference_pair(event)

        if pairs:
            for pair in pairs:
                orpo_item = {
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": pair.get("prompt", "")}
                    ],
                    "chosen": [
                        {"role": "assistant", "content": pair.get("chosen", "")}
                    ],
                    "rejected": [
                        {"role": "assistant", "content": pair.get("rejected", "")}
                    ],
                    "event_id": event.get("id"),
                    "event_name": event.get("name")
                }
                orpo_dataset.append(orpo_item)
            print(f"OK (+{len(pairs)} pairs)")
        else:
            print("SKIP")

        # Save progress every 5 events
        if (i + 1) % 5 == 0:
            with open('orpo_dataset.json', 'w', encoding='utf-8') as f:
                json.dump(orpo_dataset, f, ensure_ascii=False, indent=2)
            print(f"  [SAVED] {len(orpo_dataset)} pairs total")

        time.sleep(1)

    # Final save
    with open('orpo_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(orpo_dataset, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"DONE! Total: {len(orpo_dataset)} preference pairs")
    print("=" * 60)


if __name__ == '__main__':
    main()
