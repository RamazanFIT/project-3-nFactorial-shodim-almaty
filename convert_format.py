import json

# Load the alpaca format dataset
with open('sft_dataset_alpaca.json', 'r', encoding='utf-8') as f:
    alpaca_data = json.load(f)

# System prompt for the assistant
system_prompt = "Ты — дружелюбный помощник по мероприятиям в Алматы. Помогаешь пользователям найти интересные события, концерты, спектакли, выставки и развлечения в городе. Отвечай информативно, указывая даты, время, место и цены."

# Convert to OpenAI chat format
chat_format = []

for item in alpaca_data:
    chat_item = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": item["instruction"]
            },
            {
                "role": "assistant",
                "content": item["output"]
            }
        ]
    }
    chat_format.append(chat_item)

# Save the converted format
with open('sft_dataset_chat.json', 'w', encoding='utf-8') as f:
    json.dump(chat_format, f, ensure_ascii=False, indent=2)

print(f"Converted {len(chat_format)} items")
print(f"Saved to: sft_dataset_chat.json")
