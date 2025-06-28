import os
import logging
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

load_dotenv()
TOKEN = os.environ['TOKEN']


bot = Bot(TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    user_name = message.from_user.first_name

    await message.answer(f"Здравствуйте, {user_name}! Я бот, который помогает искать российские законы, применимые к вашей ситуации. \
                     Опишите ситуацию естественным языком, и я выдам вам список наиболее подходящих к ней законов.")
    
@dp.message()  # Handles all other messages after /start
async def handle_user_message(message: types.Message):
    query = message.text

    closest = ['a', 'b', 'very\nlong\nlaw\naaa\na'] #как-то находим самые близкие к запросу документы
    extra_text = ['Да', "Нет", "Наверное"] #как-то обрабатываем LLM


    response = '\n\n'.join([f"**>{closest[i].replace('\n', '\n>')}||\nБолее простым языком:\n{extra_text[i]}" for i in range(len(closest))])
    await message.answer(response, parse_mode='MarkdownV2')


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())