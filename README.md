# Qwen2.5-3B Sxodim Almaty 🎭

Fine-tuned модель для рекомендаций мероприятий в Алматы с использованием QLoRA + ORPO.

## Links

- **Model**: [huggingface.co/rsyrlybay/qwen2.5-3b-sxodim-almaty](https://huggingface.co/rsyrlybay/qwen2.5-3b-sxodim-almaty)
- **Training Notebook**: [Google Colab](https://colab.research.google.com/drive/1mVQa-dxBqTnVnDGmB5aD8FVps9q8bhmb?usp=sharing)
- **Data Source**: [sxodim.com](https://sxodim.com)

## Описание

Ассистент по мероприятиям в Алматы, обученный на реальных данных о концертах, спектаклях, выставках и развлечениях. Отвечает дружелюбно и по-человечески, как будто советует друг.

## Использование

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("rsyrlybay/qwen2.5-3b-sxodim-almaty")
tokenizer = AutoTokenizer.from_pretrained("rsyrlybay/qwen2.5-3b-sxodim-almaty")

messages = [
    {"role": "system", "content": "Ты — дружелюбный помощник по мероприятиям в Алматы."},
    {"role": "user", "content": "Куда сходить на выходных?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Pipeline обучения

```
Qwen2.5-3B (base)
      ↓
[SFT Training] ← sft_dataset_chat.json (1443 Q&A пар)
      ↓
SFT Model (знает факты о мероприятиях)
      ↓
[ORPO Training] ← orpo_dataset.json (~500 preference пар)
      ↓
Final Model (дружелюбный стиль ответов)
```

## Детали обучения

| Параметр | Значение |
|----------|----------|
| Base model | Qwen/Qwen2.5-3B |
| Method | QLoRA + ORPO |
| LoRA r | 32 |
| LoRA alpha | 64 |
| SFT Dataset | 1443 Q&A пар |
| ORPO Dataset | ~500 preference пар |
| Epochs | 3 |
| Learning Rate | 2e-4 |

## Данные

Собрано с [sxodim.com](https://sxodim.com):
- **104 мероприятия** в Алматы
- Категории: концерты, стендапы, спектакли, мюзиклы, выставки, детские мероприятия
- Информация: название, адрес, цена, даты, описание

## Структура проекта

```
├── scrape_sxodim.py          # Скрапер данных с sxodim.com
├── generate_sft.py           # Генерация SFT датасета
├── generate_orpo_dataset.py  # Генерация ORPO preference pairs
├── convert_format.py         # Конвертация в chat format
├── train.ipynb               # Notebook для обучения (Colab)
├── sxodim_data.json          # Сырые данные (104 мероприятия)
├── sft_dataset_chat.json     # SFT датасет (1443 пар)
└── orpo_dataset.json         # ORPO датасет (~500 пар)
```

## Примеры вопросов

- Какие концерты будут в эти выходные?
- Где можно посмотреть стендап в Алматы?
- Посоветуй куда сходить с детьми
- Сколько стоят билеты на мюзикл?
- Что интересного в Punch Stand Up Club?

## Особенности модели

- **Дружелюбный стиль** — отвечает живо, с эмоциями
- **ORPO обучение** — предпочитает человечные ответы формальным
- **Локальные знания** — адреса, цены, расписание мероприятий Алматы

## Автор

nFactorial Incubator 2025

## Лицензия

Apache 2.0
# project-3-nFactorial-shodim-almaty
