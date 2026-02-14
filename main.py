"""
DeFi Position Calculator - Production
CoinMarketCap API + Manual Input
Optimized for Railway
"""

import asyncio
import os
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
from aiogram import Bot, Dispatcher, types, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.strategy import FSMStrategy
import aiohttp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("BOT_TOKEN not set")

CMC_API_KEY = os.getenv("COINMARKETCAP_API_KEY", "")

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher(storage=MemoryStorage(), fsm_strategy=FSMStrategy.USER_IN_CHAT)

class CoinMarketCapAPI:
    BASE_URL = "https://pro-api.coinmarketcap.com/v1"
    
    def __init__(self, api_key: str = "", cache_ttl: int = 300):
        self._api_key = api_key
        self._cache: Dict[str, Tuple[float, datetime]] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl)
        self._session: Optional[aiohttp.ClientSession] = None
        self._stats = {"total": 0, "success": 0, "fail": 0, "cache": 0}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def is_available(self) -> bool:
        return bool(self._api_key)
    
    def _get_from_cache(self, symbol: str) -> Optional[float]:
        if symbol in self._cache:
            price, timestamp = self._cache[symbol]
            if datetime.now() - timestamp < self._cache_ttl:
                self._stats["cache"] += 1
                return price
        return None
    
    async def get_price_usd(self, symbol: str) -> Optional[float]:
        if not self.is_available():
            return None
        
        self._stats["total"] += 1
        symbol = symbol.upper().strip()
        
        cached = self._get_from_cache(symbol)
        if cached:
            return cached
        
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.BASE_URL}/cryptocurrency/quotes/latest",
                headers={"X-CMC_PRO_API_KEY": self._api_key, "Accept": "application/json"},
                params={"symbol": symbol}
            ) as resp:
                if resp.status != 200:
                    self._stats["fail"] += 1
                    return None
                
                data = await resp.json()
                if "data" not in data or symbol not in data["data"]:
                    self._stats["fail"] += 1
                    return None
                
                price = float(data["data"][symbol]["quote"]["USD"]["price"])
                self._stats["success"] += 1
                self._cache[symbol] = (price, datetime.now())
                logger.info(f"CMC {symbol}: ${price:,.2f}")
                return price
        except Exception as e:
            self._stats["fail"] += 1
            logger.error(f"CMC error for {symbol}: {e}")
            return None
    
    def get_stats(self) -> dict:
        total = self._stats['total']
        return {
            **self._stats,
            "success_rate": f"{self._stats['success']/total*100:.1f}%" if total > 0 else "0%",
            "cache_rate": f"{self._stats['cache']/total*100:.1f}%" if total > 0 else "0%"
        }

cmc = CoinMarketCapAPI(api_key=CMC_API_KEY, cache_ttl=300)

class Calc(StatesGroup):
    supply_ticker = State()
    borrow_ticker = State()
    supply_amount = State()
    supply_price = State()
    max_ltv = State()
    lt = State()
    mode = State()
    ltv = State()
    borrow = State()

mode_kb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🔢 By LTV", callback_data="mode_ltv")],
    [InlineKeyboardButton(text="💵 By amount", callback_data="mode_borrow")]
])

def validate_number(text: str, min_val: float = 0, max_val: Optional[float] = None) -> Tuple[bool, float, str]:
    try:
        value = float(text.replace(",", ".").strip())
        if value <= min_val:
            return False, 0, f"Must be > {min_val}"
        if max_val and value > max_val:
            return False, 0, f"Must be ≤ {max_val}"
        return True, value, ""
    except:
        return False, 0, "Enter a number"

def validate_ticker(text: str) -> Tuple[bool, str, str]:
    ticker = text.upper().strip()
    if len(ticker) > 10:
        return False, "", "Max 10 chars"
    if not ticker.isalnum():
        return False, "", "Letters/numbers only"
    return True, ticker, ""

def format_currency(value: float) -> str:
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:.2f}"

def format_price(price: float) -> str:
    if price >= 1:
        return f"${price:,.2f}"
    elif price >= 0.01:
        return f"${price:.4f}"
    elif price >= 0.0001:
        return f"${price:.6f}"
    return f"${price:.8f}"

def calc_hf(coll: float, lt: float, borrow: float) -> float:
    return (coll * lt) / borrow if borrow > 0 else float('inf')

def calc_liq(borrow: float, supply: float, lt: float) -> float:
    return borrow / (supply * lt) if (supply * lt) > 0 else 0

def get_status(hf: float) -> Tuple[str, str]:
    if hf <= 1.0:
        return "🔴 LIQUIDATION", "🔴"
    elif hf < 1.3:
        return "🟡 WARNING", "🟡"
    elif hf < 2.0:
        return "🟢 SAFE", "🟢"
    return "🔵 VERY SAFE", "🔵"

def build_result(data: dict, c: dict) -> str:
    status, emoji = get_status(c['hf'])
    source = "CMC" if data.get('price_source') == 'cmc' else "manual"
    
    result = (
        f"<b>{emoji} POSITION</b>\n"
        f"Status: <b>{status}</b>\n\n"
        f"<b>💎 COLLATERAL:</b>\n"
        f"• {data['supply_ticker']}: {c['supply']:.6f}\n"
        f"• Price: {format_price(c['price'])} ({source})\n"
        f"• Value: <b>{format_currency(c['coll'])}</b>\n\n"
        f"<b>💰 BORROW:</b>\n"
        f"• {data['borrow_ticker']}: <b>{format_currency(c['borrow'])}</b>\n\n"
        f"<b>⚙️ PARAMS:</b>\n"
        f"• Max LTV: {c['max_ltv']:.0f}%\n"
        f"• LT: {c['lt']*100:.1f}%\n"
        f"• Current LTV: <b>{c['ltv']:.2f}%</b>\n\n"
        f"<b>📊 RISKS:</b>\n"
        f"• HF: <b>{c['hf']:.2f if c['hf'] != float('inf') else '∞'}</b>\n"
        f"• Liquidation: <b>{format_price(c['liq'])}</b>\n"
        f"• Buffer: <b>{c['buffer']:.1f}%</b>\n"
        f"• Max borrow: {format_currency(c['max_borrow'])}\n\n"
        f"<b>📉 SCENARIOS:</b>\n"
    )
    
    for drop, hf in c['scenarios']:
        new_price = c['price'] * (1 - drop / 100)
        result += f"• -{drop}% ({format_price(new_price)}) → HF: {hf:.2f}\n"
    
    if c['hf'] < 1.3:
        result += "\n<b>⚠️ WARNINGS:</b>\n• Increase collateral\n• Reduce borrow\n• Monitor price"
    
    return result

@dp.message(Command("start"))
async def start_cmd(msg: types.Message, state: FSMContext):
    await state.clear()
    cmc_status = "✅" if cmc.is_available() else "❌"
    await msg.answer(
        f"🤖 <b>DeFi Calculator</b>\n\n"
        f"<b>Price sources:</b>\n{cmc_status} CoinMarketCap\n✅ Manual\n\n"
        f"Enter <b>collateral ticker</b>:"
    )
    await state.set_state(Calc.supply_ticker)

@dp.message(Command("reset"))
async def reset_cmd(msg: types.Message, state: FSMContext):
    await state.clear()
    await msg.answer("✅ Reset. /start")

@dp.message(Command("stats"))
async def stats_cmd(msg: types.Message):
    stats = cmc.get_stats()
    await msg.answer(
        f"<b>📊 CMC Stats</b>\n\n"
        f"Requests: {stats['total']}\n"
        f"Success: {stats['success']}\n"
        f"Cached: {stats['cache']}\n"
        f"Failed: {stats['fail']}\n"
        f"Success rate: {stats['success_rate']}\n"
        f"Cache rate: {stats['cache_rate']}"
    )

@dp.message(Calc.supply_ticker)
async def process_supply_ticker(msg: types.Message, state: FSMContext):
    valid, ticker, error = validate_ticker(msg.text)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    await state.update_data(supply_ticker=ticker)
    await msg.answer(f"✅ Collateral: <b>{ticker}</b>\n\nEnter borrow ticker:")
    await state.set_state(Calc.borrow_ticker)

@dp.message(Calc.borrow_ticker)
async def process_borrow_ticker(msg: types.Message, state: FSMContext):
    valid, ticker, error = validate_ticker(msg.text)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    await state.update_data(borrow_ticker=ticker)
    data = await state.get_data()
    await msg.answer(f"✅ Borrow: <b>{ticker}</b>\n\nEnter {data['supply_ticker']} amount:")
    await state.set_state(Calc.supply_amount)

@dp.message(Calc.supply_amount)
async def process_supply_amount(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0.000001)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    
    await state.update_data(supply_amount=value)
    data = await state.get_data()
    ticker = data['supply_ticker']
    
    if cmc.is_available():
        await msg.answer(f"✅ Amount: {value:.6f}\n\n⏳ Getting price...")
        price = await cmc.get_price_usd(ticker)
        
        if price:
            await state.update_data(supply_price=price, price_source='cmc')
            coll = value * price
            await msg.answer(
                f"✅ Price (CMC): <b>{format_price(price)}</b>\n"
                f"💰 Collateral: <b>{format_currency(coll)}</b>\n\n"
                f"Enter <b>Maximum LTV</b> in %:"
            )
            await state.set_state(Calc.max_ltv)
            return
    
    await msg.answer(f"✅ Amount: {value:.6f}\n\nEnter <b>{ticker} price</b> in USD:")
    await state.set_state(Calc.supply_price)

@dp.message(Calc.supply_price)
async def process_supply_price(msg: types.Message, state: FSMContext):
    valid, price, error = validate_number(msg.text, min_val=0.000001)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    
    await state.update_data(supply_price=price, price_source='manual')
    data = await state.get_data()
    coll = data['supply_amount'] * price
    
    await msg.answer(
        f"✅ Price: <b>{format_price(price)}</b>\n"
        f"💰 Collateral: <b>{format_currency(coll)}</b>\n\n"
        f"Enter <b>Maximum LTV</b> in %:"
    )
    await state.set_state(Calc.max_ltv)

@dp.message(Calc.max_ltv)
async def process_max_ltv(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    
    await state.update_data(max_ltv=value / 100)
    data = await state.get_data()
    coll = data['supply_amount'] * data['supply_price']
    max_borrow = coll * (value / 100)
    
    await msg.answer(
        f"✅ <b>Max LTV: {value}%</b>\n"
        f"💰 Max borrow: <b>{format_currency(max_borrow)}</b>\n\n"
        f"Enter <b>Liquidation Threshold</b> in %:"
    )
    await state.set_state(Calc.lt)

@dp.message(Calc.lt)
async def process_lt(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    
    data = await state.get_data()
    max_ltv = data['max_ltv'] * 100
    
    if value < max_ltv:
        await msg.answer(f"❌ LT ({value}%) < Max LTV ({max_ltv:.0f}%)")
        return
    
    await state.update_data(lt=value / 100)
    await msg.answer(f"✅ <b>LT: {value}%</b>\n\nChoose mode:", reply_markup=mode_kb)
    await state.set_state(Calc.mode)

@dp.callback_query(F.data.startswith("mode_"))
async def process_mode(cb: types.CallbackQuery, state: FSMContext):
    await cb.answer()
    data = await state.get_data()
    coll = data['supply_amount'] * data['supply_price']
    max_ltv = data['max_ltv']
    
    await state.update_data(mode=cb.data)
    
    if cb.data == "mode_ltv":
        await cb.message.edit_text(
            f"<b>🔢 Mode: By LTV</b>\n\n"
            f"Collateral: {format_currency(coll)}\n"
            f"Max LTV: {max_ltv * 100:.0f}%\n\n"
            f"Enter <b>LTV</b> in %:"
        )
        await state.set_state(Calc.ltv)
    else:
        max_borrow = coll * max_ltv
        await cb.message.edit_text(
            f"<b>💵 Mode: By amount</b>\n\n"
            f"Collateral: {format_currency(coll)}\n"
            f"Max borrow: <b>{format_currency(max_borrow)}</b>\n\n"
            f"Enter borrow amount in USD:"
        )
        await state.set_state(Calc.borrow)

@dp.message(Calc.ltv)
async def process_ltv(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    
    data = await state.get_data()
    max_ltv = data['max_ltv'] * 100
    
    if value > max_ltv:
        await msg.answer(f"❌ LTV ({value}%) > Max LTV ({max_ltv:.0f}%)")
        return
    
    await state.update_data(ltv=value / 100)
    await calculate(msg, state)

@dp.message(Calc.borrow)
async def process_borrow(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    
    data = await state.get_data()
    coll = data['supply_amount'] * data['supply_price']
    max_borrow = coll * data['max_ltv']
    
    if value > max_borrow:
        await msg.answer(f"❌ {format_currency(value)} > max {format_currency(max_borrow)}")
        return
    
    await state.update_data(borrow=value)
    await calculate(msg, state)

async def calculate(msg: types.Message, state: FSMContext):
    try:
        data = await state.get_data()
        supply_amt = data['supply_amount']
        price = data['supply_price']
        lt = data['lt']
        max_ltv = data['max_ltv']
        
        coll = supply_amt * price
        
        if data['mode'] == "mode_ltv":
            borrow = coll * data['ltv']
            ltv = data['ltv']
        else:
            borrow = data['borrow']
            ltv = borrow / coll if coll > 0 else 0
        
        hf = calc_hf(coll, lt, borrow)
        liq = calc_liq(borrow, supply_amt, lt)
        buffer = ((price - liq) / price) * 100 if price > 0 else 0
        
        scenarios = []
        for drop in [10, 20, 30]:
            new_coll = supply_amt * price * (1 - drop / 100)
            scenarios.append((drop, calc_hf(new_coll, lt, borrow)))
        
        c = {
            'supply': supply_amt,
            'price': price,
            'coll': coll,
            'borrow': borrow,
            'ltv': ltv * 100,
            'max_ltv': max_ltv * 100,
            'lt': lt,
            'hf': hf,
            'liq': liq,
            'buffer': buffer,
            'max_borrow': coll * max_ltv,
            'scenarios': scenarios
        }
        
        await msg.answer("⏳ Calculating...")
        await msg.answer(build_result(data, c))
        await msg.answer("✅ Done!\n\n/start - new")
        await state.clear()
        
    except Exception as e:
        logger.error(f"Calc error: {e}")
        await msg.answer(f"❌ Error\n\n/start")
        await state.clear()

@dp.message()
async def fallback(msg: types.Message, state: FSMContext):
    if await state.get_state():
        await msg.answer("⚠️ Follow instructions or /reset")
    else:
        await msg.answer("👋 /start")

@dp.error()
async def error_handler(update: types.Update, exception: Exception):
    logger.error(f"Error: {exception}", exc_info=True)
    return True

async def on_startup():
    logger.info("🚀 DeFi Calculator v1.0")
    bot_info = await bot.get_me()
    logger.info(f"✅ Bot: @{bot_info.username}")
    
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("✅ Webhook deleted")
    except:
        pass
    
    if cmc.is_available():
        logger.info("✅ CoinMarketCap configured")
        test = await cmc.get_price_usd("BTC")
        if test:
            logger.info(f"✅ Test: BTC=${test:,.2f}")
    else:
        logger.info("⚠️  CoinMarketCap not configured")

async def on_shutdown():
    await cmc.close()
    await bot.session.close()
    logger.info("👋 Stopped")

async def main():
    try:
        await on_startup()
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types(), close_timeout=10, timeout=30)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        await on_shutdown()

if __name__ == "__main__":
    asyncio.run(main())
