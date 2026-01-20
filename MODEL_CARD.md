---
license: apache-2.0
language:
  - ru
base_model: Qwen/Qwen2.5-3B
tags:
  - almaty
  - events
  - qlora
  - orpo
  - fine-tuned
  - text-generation
  - assistant
  - sxodim
pipeline_tag: text-generation
datasets:
  - custom
---

# Qwen2.5-3B Sxodim Almaty 🎭

Модель fine-tuned на данных о мероприятиях Алматы с [sxodim.com](https://sxodim.com) с использованием QLoRA + ORPO.

## Описание

Дружелюбный ассистент по мероприятиям в Алматы. Помогает найти концерты, спектакли, выставки и развлечения. Отвечает живо и по-человечески, как друг.

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

## Training Pipeline

```
Qwen2.5-3B (base)
      ↓
[SFT Training] ← 1443 Q&A пар
      ↓
[ORPO Training] ← ~500 preference пар
      ↓
Final Model
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
- Концерты, стендапы, спектакли, мюзиклы, выставки, детские мероприятия
- Адреса, цены, даты, описания

## Примеры

**User:** Какие концерты будут в эти выходные?

**Assistant:** О, в эти выходные много крутого! Советую заглянуть на стендап в Punch Stand Up Club — там всегда угар. Если хочешь что-то музыкальное, посмотри афишу EverJazz, у них часто бывают джазовые вечера. А для детей есть мюзиклы в разных театрах города!

---

**User:** Где посмотреть стендап в Алматы?

**Assistant:** Самое топовое место — это Punch Stand Up Club! Там постоянно выступают и местные комики, и приезжие звёзды. Атмосфера классная, зал уютный. Билеты обычно от 5000 тенге, но на известных комиков может быть дороже. Очень рекомендую!

## Links

- **Training Notebook:** [Google Colab](https://colab.research.google.com/drive/1mVQa-dxBqTnVnDGmB5aD8FVps9q8bhmb?usp=sharing)
- **Data Source:** [sxodim.com](https://sxodim.com)

## Автор

nFactorial Incubator 2025

## Лицензия

Apache 2.0
