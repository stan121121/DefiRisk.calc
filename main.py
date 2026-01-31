Python 3.14.2 (tags/v3.14.2:df79316, Dec  5 2025, 17:18:21) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> import asyncio
... from aiogram import Bot, Dispatcher, types
... from aiogram.filters import Command
... from aiogram.fsm.state import StatesGroup, State
... from aiogram.fsm.context import FSMContext
... from aiogram.fsm.storage.memory import MemoryStorage
... from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
... TOKEN = TOKEN = "PASTE_YOUR_BOT_TOKEN"
... 
... import os
... bot = os.getenv(token=TOKEN)
... dp = Dispatcher(storage=MemoryStorage())
... 
... # ---------- STATES ----------
... class Calc(StatesGroup):
...     supply_ticker = State()
...     borrow_ticker = State()
...     supply_amount = State()
...     supply_price = State()
...     mode = State()
...     ltv = State()
...     borrow = State()
...     lt = State()
...     max_ltv = State()
... 
... # ---------- KEYBOARD ----------
... mode_kb = InlineKeyboardMarkup(inline_keyboard=[
...     [InlineKeyboardButton(text="🔢 By LTV", callback_data="mode_ltv")],
...     [InlineKeyboardButton(text="💵 By Borrow", callback_data="mode_borrow")]
... ])
... 
... # ---------- START ----------
... @dp.message(Command("start"))
... async def start(msg: types.Message, state: FSMContext):
...     await state.clear()
...     await msg.answer("Введите Supply ticker (ETH, SOL, BTC):")
...     await state.set_state(Calc.supply_ticker)

@dp.message(Command("reset"))
async def reset(msg: types.Message, state: FSMContext):
    await state.clear()
    await msg.answer("Сброшено. /start")

# ---------- FLOW ----------
@dp.message(Calc.supply_ticker)
async def supply_ticker(msg, state):
    await state.update_data(supply_ticker=msg.text.upper())
    await msg.answer("Введите Borrow ticker (USDC, DAI):")
    await state.set_state(Calc.borrow_ticker)

@dp.message(Calc.borrow_ticker)
async def borrow_ticker(msg, state):
    await state.update_data(borrow_ticker=msg.text.upper())
    await msg.answer("Введите количество Supply:")
    await state.set_state(Calc.supply_amount)

@dp.message(Calc.supply_amount)
async def supply_amount(msg, state):
    await state.update_data(supply_amount=float(msg.text))
    await msg.answer("Введите цену Supply (USD):")
    await state.set_state(Calc.supply_price)

@dp.message(Calc.supply_price)
async def supply_price(msg, state):
    await state.update_data(supply_price=float(msg.text))
    await msg.answer("Выберите режим расчёта:", reply_markup=mode_kb)
    await state.set_state(Calc.mode)

@dp.callback_query(lambda c: c.data.startswith("mode_"))
async def mode_select(cb: types.CallbackQuery, state: FSMContext):
    mode = cb.data
    await cb.answer()
    await state.update_data(mode=mode)

    if mode == "mode_ltv":
        await cb.message.answer("Введите LTV (%):")
        await state.set_state(Calc.ltv)
    else:
        await cb.message.answer("Введите Borrow amount:")
        await state.set_state(Calc.borrow)

@dp.message(Calc.ltv)
async def input_ltv(msg, state):
    await state.update_data(ltv=float(msg.text) / 100)
    await msg.answer("Введите Liquidation Threshold (%):")
    await state.set_state(Calc.lt)

@dp.message(Calc.borrow)
async def input_borrow(msg, state):
    await state.update_data(borrow=float(msg.text))
    await msg.answer("Введите Liquidation Threshold (%):")
    await state.set_state(Calc.lt)

@dp.message(Calc.lt)
async def input_lt(msg, state):
    await state.update_data(lt=float(msg.text) / 100)
    await msg.answer("Введите Max LTV (%):")
    await state.set_state(Calc.max_ltv)

# ---------- CALC ----------
@dp.message(Calc.max_ltv)
async def calculate(msg, state):
    data = await state.get_data()

    supply_amt = data["supply_amount"]
    price = data["supply_price"]
    lt = data["lt"]
    max_ltv = float(msg.text) / 100

    collateral = supply_amt * price

    if data["mode"] == "mode_ltv":
        ltv = data["ltv"]
        borrow = collateral * ltv
    else:
        borrow = data["borrow"]
        ltv = borrow / collateral

    hf = (collateral * lt) / borrow
    liq_price = borrow / (supply_amt * lt)
    max_borrow = collateral * max_ltv
    buffer = (price - liq_price) / price

    # scenarios
    price_10 = price * 0.9
    price_20 = price * 0.8
    hf_10 = (supply_amt * price_10 * lt) / borrow
    hf_20 = (supply_amt * price_20 * lt) / borrow

    status = "🟢 SAFE"
    if hf <= 1:
        status = "🔴 LIQUIDATION"
    elif hf < 1.3:
        status = "🟡 WARNING"

    await msg.answer(
        f"📊 Position summary\n\n"
        f"Supply: {supply_amt} {data['supply_ticker']}\n"
        f"Price: ${price:.2f}\n"
        f"Collateral: ${collateral:.2f}\n\n"
        f"Borrow: ${borrow:.2f} {data['borrow_ticker']}\n"
        f"LTV: {ltv*100:.2f}%\n\n"
        f"Health Factor: {hf:.2f} {status}\n"
        f"Liquidation price: ${liq_price:.2f}\n"
        f"Max borrow: ${max_borrow:.2f}\n"
        f"Buffer: {buffer*100:.2f}%\n\n"
        f"📉 Scenarios:\n"
        f"-10% price → HF {hf_10:.2f}\n"
        f"-20% price → HF {hf_20:.2f}"
    )

    await state.clear()

# ---------- RUN ----------
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())


