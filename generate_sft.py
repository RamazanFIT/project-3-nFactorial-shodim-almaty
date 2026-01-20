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

def generate_qa_pairs(event):
    """Generate question-answer pairs for an event using Gemini"""

    # Prepare event context
    event_context = f"""
Название: {event.get('name', 'Не указано')}
Категория: {event.get('category', 'Не указано')}
Город: {event.get('city', 'Алматы')}
Адрес: {event.get('address', 'Не указано')}
Цена билетов: {event.get('ticket_price', 'Не указано')}
Описание: {event.get('description', 'Не указано')}
URL: {event.get('url', '')}

Полное содержимое:
{event.get('markdown_content', event.get('content_html', 'Нет информации'))[:4000]}
"""

    prompt = f"""На основе информации о мероприятии в Алматы, создай 12-15 разнообразных пар "вопрос-ответ" на русском языке.

Информация о мероприятии:
{event_context}

Требования к вопросам:
1. Вопросы должны быть естественными, как если бы их задавал пользователь в чате
2. Включи разные типы вопросов:
   - Общие вопросы ("Что посоветуешь на выходные?", "Куда сходить сегодня?")
   - Конкретные вопросы о мероприятии ("Когда начало концерта X?", "Сколько стоят билеты на Y?")
   - Вопросы о месте проведения ("Где находится Z?", "Как добраться до X?")
   - Рекомендательные вопросы ("Что интересного для детей?", "Какие концерты есть в эти выходные?")

3. Ответы должны быть:
   - Информативными и полезными
   - Дружелюбными в тоне
   - Содержать конкретные данные из мероприятия (дата, время, цена, адрес)

Формат ответа - JSON массив:
[
  {{"question": "вопрос пользователя", "answer": "ответ ассистента"}},
  ...
]

Отвечай ТОЛЬКО JSON массивом, без дополнительного текста."""

    try:
        response = client.chat.completions.create(
            model="google/gemini-3-flash-preview",
            messages=[{"role": "user", "content": prompt}],
            extra_body={
                "reasoning": {"enabled": True},
                "provider": {"allow_fallbacks": False, "only": ["google-ai-studio"]}
            }
        )

        content = response.choices[0].message.content

        # Clean up response - remove markdown code blocks if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        qa_pairs = json.loads(content)
        return qa_pairs

    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Raw content: {content[:200]}...")
        return []
    except Exception as e:
        print(f"  API error: {e}")
        return []

def main():
    # Load events data
    with open('sxodim_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    events = data['events']

    print("=" * 60)
    print("SFT DATASET GENERATOR")
    print("=" * 60)
    print(f"Total events: {len(events)}")
    print("=" * 60)

    sft_dataset = []

    for i, event in enumerate(events):
        name = event.get('name', 'Unknown')[:45]
        print(f"[{i+1}/{len(events)}] {name}...", end=" ", flush=True)

        qa_pairs = generate_qa_pairs(event)

        if qa_pairs:
            for qa in qa_pairs:
                sft_dataset.append({
                    "instruction": qa.get("question", ""),
                    "input": "",
                    "output": qa.get("answer", ""),
                    "event_id": event.get("id"),
                    "event_name": event.get("name"),
                    "category": event.get("category")
                })
            print(f"OK (+{len(qa_pairs)} pairs)")
        else:
            print("SKIP")

        # Save progress every 10 events
        if (i + 1) % 10 == 0:
            with open('sft_dataset.json', 'w', encoding='utf-8') as f:
                json.dump(sft_dataset, f, ensure_ascii=False, indent=2)
            print(f"  [SAVED] {len(sft_dataset)} pairs total")

        time.sleep(1)  # Rate limiting

    # Final save
    with open('sft_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(sft_dataset, f, ensure_ascii=False, indent=2)

    # Also save in Alpaca format
    alpaca_format = []
    for item in sft_dataset:
        alpaca_format.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"]
        })

    with open('sft_dataset_alpaca.json', 'w', encoding='utf-8') as f:
        json.dump(alpaca_format, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"DONE! Generated {len(sft_dataset)} QA pairs")
    print(f"Saved to: sft_dataset.json")
    print(f"Saved to: sft_dataset_alpaca.json (Alpaca format)")
    print("=" * 60)

if __name__ == '__main__':
    main()
