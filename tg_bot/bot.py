import os
import logging
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import ollama

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
from collections import defaultdict

load_dotenv()
TOKEN = os.environ['TOKEN']

model = SentenceTransformer('cointegrated/rubert-tiny2')
index = faiss.read_index("../embeddings/embeddings.index")

chunk_files = sorted(Path('..').glob('dataset/preproc_data*.parquet'))
dfs = [pd.read_parquet(f) for f in chunk_files]
df = pd.concat(dfs, ignore_index=True)

df[['classifier_code', 'classifier_name']] = df['classifierByIPS'].str.split('$', n=1, expand=True)
df['classifier_level2'] = df['classifier_code'].str.extract(r'^(\d{3}\.\d{3})')


template = '''Контекст: {context}
Исходя из контекста, ответьте на следующий вопрос:
Вопрос: {query}
Предоставьте ответ только на основе предоставленного контекста, без использования общих знаний. Ответ должен быть непосредственно взят из предоставленного контекста.
Пожалуйста, исправьте грамматические ошибки для улучшения читаемости.
Если в контексте нет информации, достаточной для ответа на вопрос, укажите, что ответ отсутствует в данном контексте.
'''


bot = Bot(TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)


def generate_llm_response(query, top_results, template, model_name="gemma3:1b"):
    # generate context of top 5
    context = ""
    sources = []
    
    for i, res in enumerate(top_results, 1):
        context += f"[[Источник {i}]]\n{res['text']}\n\n"
        sources.append({
            "index": res['index'],
            "similarity": res['similarity'],
            "classifier": res['classifier'],
            'heading': res['heading'],
            'text': res['text']
        })
        
    prompt = template.format(context=context, query=query)
    
    # generate responce from ollama 
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={
            "temperature": 0.3,
            "num_ctx": 10000 # context size
        }
    )
    
    answer = response['response']
    
    answer += "\n\n---\nRetrieval laws:\n"
    for src in sources:
        answer += (f"- [ID: {src['index']}] cos similarity: {src['similarity']:.3f},\n"
                  f"classifier code: {src['classifier']}, 'low heading: {src['heading']}\n")

    return answer

def RAG(query):
    query_vec = model.encode([query]) # encode
    query_vec = query_vec / np.linalg.norm(query_vec) # normalyze

    faiss.normalize_L2(query_vec)

    k = 50
    # find the closest
    similarities, indices = index.search(query_vec, k)

    # filter resilt from top 50 to top 5
    results = []
    for idx, sim in zip(indices[0], similarities[0]):
        if idx == -1:  # if idx not exist, skip
            continue
        
        row = df.iloc[idx]
        classifier = str(row['classifier_level2']).strip() if pd.notna(row['classifier_level2']) else "UNKNOWN"
        
        # add if not UNKNOWN
        if classifier != "UNKNOWN":
            results.append({
                'index': idx,
                'similarity': sim,
                'classifier': row['classifier_code'],
                'classifier_level2': classifier,
                'text': row['textIPS'],
                'heading': row['headingIPS']
            })

    # group by classifier_level2 (2 numbers from classifier_code)
    grouped = defaultdict(list)
    for res in results:
        grouped[res['classifier_level2']].append(res)

    top_results = []
    # take from each grouped 1 with max similarity
    if len(grouped) >= 5:
        for cls in sorted(grouped.keys(), key=lambda x: max(y['similarity'] for y in grouped[x]), reverse=True)[:5]:
            best_in_cls = max(grouped[cls], key=lambda x: x['similarity'])
            top_results.append(best_in_cls)

    # take from each grouped 2 with max similarity
    else:
        candidates = []
        for cls, items in grouped.items():
            items_sorted = sorted(items, key=lambda x: -x['similarity'])
            candidates.extend(items_sorted[:2])
        # return only top 5
        top_results = sorted(candidates, key=lambda x: -x['similarity'])[:5]

    print("Топ-5 результатов с учетом классов:")
    for i, res in enumerate(top_results, 1):
        print(f"{i}. Схожесть: {res['similarity']:.3f} | Класс: {res['classifier_level2']}")
        print(f"Текст: {res['text']}")
    return top_results

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    user_name = message.from_user.first_name

    await message.answer(f"Здравствуйте, {user_name}! Я бот, который помогает искать российские законы, применимые к вашей ситуации.\nОпишите ситуацию естественным языком, и я выдам вам список наиболее подходящих к ней законов.")
    
@dp.message()  # Handles all other messages after /start
async def handle_user_message(message: types.Message):
    query = message.text

    top_res = RAG(query)
    await message.answer('Пока ждете ответа, ознакомьтесь с документами, которые могут помочь в вашей ситуации:')
    for i, res in enumerate(top_res, 1):
        text = f"{i}. классификатор: {res['classifier']}\nТекст документа: {res['text'][:2000]}..."
        await message.answer(text)

    await message.answer('Идет поиск по предложенной информации и правовым документам... (около 2 мин)')
    
    answer = generate_llm_response(query, top_res, template, model_name="gemma3:1b")
    
    await message.answer(answer)


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())