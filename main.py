"""
=============================================================================
DeFi Position Calculator Bot - Production v2.4
Optimized for Railway with CryptoRank v1 API
=============================================================================
"""

import asyncio
import os
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.strategy import FSMStrategy
from typing import Tuple, Optional, Dict
import aiohttp
from datetime import datetime, timedelta
from collections import deque

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("❌ BOT_TOKEN not set in environment variables")

CRYPTORANK_API_KEY = os.getenv("CRYPTORANK_API_KEY", "")

# =============================================================================
# PRICE FETCHERS
# =============================================================================

class CryptoRankPriceFetcher:
    """CryptoRank v1 API price fetcher"""
    
    BASE_URL = "https://api.cryptorank.io/v1/currencies"
    
    def __init__(self, api_key: str = ""):
        self._api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._stats = {"total": 0, "success": 0, "fail": 0}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def is_available(self) -> bool:
        return bool(self._api_key)
    
    async def get_price_usd(self, symbol: str) -> Optional[float]:
        """Получить цену через CryptoRank v1 API"""
        if not self.is_available():
            return None
        
        self._stats["total"] += 1
        symbol = symbol.upper().strip()
        
        try:
            session = await self._get_session()
            # v1 API: GET /v1/currencies/{symbol}
            url = f"{self.BASE_URL}/{symbol}"
            headers = {"api-key": self._api_key}
            
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    self._stats["fail"] += 1
                    if resp.status == 404:
                        logger.info(f"CryptoRank: {symbol} not found")
                    else:
                        logger.warning(f"CryptoRank {symbol}: HTTP {resp.status}")
                    return None
                
                data = await resp.json()
                
                # v1 структура: {"data": {"price": {"USD": value}}}
                if "data" not in data:
                    self._stats["fail"] += 1
                    return None
                
                price_data = data["data"].get("price", {})
                if "USD" not in price_data:
                    self._stats["fail"] += 1
                    return None
                
                price = float(price_data["USD"])
                self._stats["success"] += 1
                logger.info(f"CryptoRank {symbol}: ${price:,.2f}")
                return price
                
        except aiohttp.ClientError as e:
            self._stats["fail"] += 1
            logger.error(f"CryptoRank {symbol} network error: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            self._stats["fail"] += 1
            logger.error(f"CryptoRank {symbol} parse error: {e}")
            return None
        except Exception as e:
            self._stats["fail"] += 1
            logger.error(f"CryptoRank {symbol} unexpected error: {e}")
            return None
    
    def get_stats(self) -> dict:
        return self._stats


class CoinGeckoPriceFetcher:
    """CoinGecko API price fetcher with caching"""
    
    COINGECKO_IDS = {
        "ETH": "ethereum", "BTC": "bitcoin", "SOL": "solana",
        "USDC": "usd-coin", "USDT": "tether", "DAI": "dai",
        "BUSD": "binance-usd", "BNB": "binancecoin", "ADA": "cardano",
        "DOT": "polkadot", "AVAX": "avalanche-2", "MATIC": "matic-network",
        "LINK": "chainlink", "UNI": "uniswap", "ATOM": "cosmos",
        "XRP": "ripple", "LTC": "litecoin", "DOGE": "dogecoin",
        "SHIB": "shiba-inu", "AAVE": "aave",
    }
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, cache_ttl: int = 300):
        self._cache: Dict[str, Tuple[float, datetime]] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl)
        self._session: Optional[aiohttp.ClientSession] = None
        self._stats = {"total": 0, "cache": 0, "api": 0}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _get_from_cache(self, symbol: str) -> Optional[float]:
        if symbol in self._cache:
            price, timestamp = self._cache[symbol]
            if datetime.now() - timestamp < self._cache_ttl:
                self._stats["cache"] += 1
                return price
        return None
    
    async def get_price_usd(self, symbol: str) -> Optional[float]:
        symbol = symbol.upper().strip()
        self._stats["total"] += 1
        
        cached = self._get_from_cache(symbol)
        if cached is not None:
            return cached
        
        if symbol not in self.COINGECKO_IDS:
            return None
        
        try:
            session = await self._get_session()
            self._stats["api"] += 1
            
            async with session.get(
                f"{self.BASE_URL}/simple/price",
                params={"ids": self.COINGECKO_IDS[symbol], "vs_currencies": "usd"}
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                
                coin_id = self.COINGECKO_IDS[symbol]
                if coin_id not in data or "usd" not in data[coin_id]:
                    return None
                
                price = data[coin_id]["usd"]
                self._cache[symbol] = (price, datetime.now())
                logger.info(f"CoinGecko {symbol}: ${price:,.2f}")
                return price
        except Exception as e:
            logger.error(f"CoinGecko {symbol} error: {e}")
            return None
    
    @classmethod
    def is_supported(cls, symbol: str) -> bool:
        return symbol.upper().strip() in cls.COINGECKO_IDS


# Initialize fetchers
cryptorank = CryptoRankPriceFetcher(api_key=CRYPTORANK_API_KEY)
coingecko = CoinGeckoPriceFetcher(cache_ttl=300)

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher(storage=MemoryStorage(), fsm_strategy=FSMStrategy.USER_IN_CHAT)

# =============================================================================
# FSM STATES
# =============================================================================

class Calc(StatesGroup):
    supply_ticker = State()
    borrow_ticker = State()
    supply_amount = State()
    choose_price = State()
    supply_price_manual = State()
    max_ltv = State()
    lt = State()
    mode = State()
    ltv = State()
    borrow = State()

# =============================================================================
# KEYBOARDS
# =============================================================================

def price_choice_kb(cr_price: Optional[float], cg_price: Optional[float]):
    buttons = []
    
    # CoinGecko первым (более надежный)
    if cg_price is not None:
        price_str = f"${cg_price:,.2f}" if cg_price >= 1 else f"${cg_price:.6f}"
        buttons.append([InlineKeyboardButton(
            text=f"🦎 CoinGecko: {price_str}",
            callback_data="price_coingecko"
        )])
    
    # CryptoRank вторым
    if cr_price is not None:
        price_str = f"${cr_price:,.2f}" if cr_price >= 1 else f"${cr_price:.6f}"
        buttons.append([InlineKeyboardButton(
            text=f"✅ CryptoRank: {price_str}",
            callback_data="price_cryptorank"
        )])
    
    buttons.append([InlineKeyboardButton(
        text="✏️ Ввести вручную",
        callback_data="price_manual"
    )])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)


mode_kb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🔢 По LTV", callback_data="mode_ltv")],
    [InlineKeyboardButton(text="💵 По сумме", callback_data="mode_borrow")]
])

# =============================================================================
# UTILITIES
# =============================================================================

def validate_number(text: str, min_val: float = 0, max_val: Optional[float] = None) -> Tuple[bool, float, str]:
    try:
        value = float(text.replace(",", ".").strip())
        if value <= min_val:
            return False, 0, f"Должно быть > {min_val}"
        if max_val and value > max_val:
            return False, 0, f"Должно быть ≤ {max_val}"
        return True, value, ""
    except:
        return False, 0, "Введите число"


def validate_ticker(text: str) -> Tuple[bool, str, str]:
    ticker = text.upper().strip()
    if len(ticker) > 10:
        return False, "", "Макс. 10 символов"
    if not ticker.isalnum():
        return False, "", "Только буквы/цифры"
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


def calc_hf(collateral: float, lt: float, borrow: float) -> float:
    return (collateral * lt) / borrow if borrow > 0 else float('inf')


def calc_liq_price(borrow: float, supply: float, lt: float) -> float:
    return borrow / (supply * lt) if (supply * lt) > 0 else 0


def get_status(hf: float) -> Tuple[str, str]:
    if hf <= 1.0:
        return "🔴 ЛИКВИДАЦИЯ", "🔴"
    elif hf < 1.3:
        return "🟡 РИСК", "🟡"
    elif hf < 2.0:
        return "🟢 НОРМА", "🟢"
    return "🔵 ОТЛИЧНО", "🔵"


def build_result(data: dict, calc: dict) -> str:
    status, emoji = get_status(calc['hf'])
    
    sources = {
        "cryptorank": "CryptoRank",
        "coingecko": "CoinGecko",
        "manual": "ручной"
    }
    source = sources.get(data.get('supply_price_source', 'manual'), 'API')
    
    price_str = format_price(calc['price'])
    liq_str = format_price(calc['liq_price'])
    
    result = (
        f"<b>{emoji} РАСЧЕТ ПОЗИЦИИ</b>\n"
        f"Статус: <b>{status}</b>\n\n"
        
        f"<b>💎 ЗАЛОГ:</b>\n"
        f"• {data['supply_ticker']}: {calc['supply_amt']:.6f}\n"
        f"• Цена: {price_str} ({source})\n"
        f"• Стоимость: <b>{format_currency(calc['collateral'])}</b>\n\n"
        
        f"<b>💰 ЗАЙМ:</b>\n"
        f"• {data['borrow_ticker']}: <b>{format_currency(calc['borrow'])}</b>\n\n"
        
        f"<b>⚙️ ПАРАМЕТРЫ:</b>\n"
        f"• Max LTV: {calc['max_ltv_pct']}%\n"
        f"• LT: {calc['lt']*100:.1f}%\n"
        f"• Current LTV: <b>{calc['ltv_pct']:.2f}%</b>\n\n"
        
        f"<b>📊 РИСКИ:</b>\n"
        f"• HF: <b>{calc['hf']:.2f if calc['hf'] != float('inf') else '∞'}</b>\n"
        f"• Ликвидация: <b>{liq_str}</b>\n"
    )
    
    if data.get('supply_price_source') == 'manual':
        result += f"  <i>(при цене {price_str})</i>\n"
    
    result += (
        f"• Буфер: <b>{calc['buffer']:.1f}%</b>\n"
        f"• Макс. займ: {format_currency(calc['max_borrow'])}\n\n"
        
        f"<b>📉 СЦЕНАРИИ:</b>\n"
    )
    
    for drop, hf in calc['scenarios']:
        new_price = calc['price'] * (1 - drop / 100)
        result += f"• -{drop}% ({format_price(new_price)}) → HF: {hf:.2f}\n"
    
    if calc['hf'] < 1.3:
        result += (
            "\n<b>⚠️ ВНИМАНИЕ:</b>\n"
            "• Увеличьте залог\n"
            "• Уменьшите займ\n"
            "• Следите за ценой"
        )
    
    return result

# =============================================================================
# HANDLERS
# =============================================================================

@dp.message(Command("start"))
async def start_cmd(msg: types.Message, state: FSMContext):
    await state.clear()
    
    cr_status = "✅" if cryptorank.is_available() else "❌"
    
    await msg.answer(
        "🤖 <b>DeFi Calculator v2.4</b>\n\n"
        f"<b>Источники цен:</b>\n"
        f"{cr_status} CryptoRank v1\n"
        f"✅ CoinGecko\n"
        f"✅ Ручной ввод\n\n"
        "Введите <b>тикер залога</b>:"
    )
    await state.set_state(Calc.supply_ticker)


@dp.message(Command("reset"))
async def reset_cmd(msg: types.Message, state: FSMContext):
    await state.clear()
    await msg.answer("✅ Сброс. /start для нового расчета")


@dp.message(Calc.supply_ticker)
async def process_supply_ticker(msg: types.Message, state: FSMContext):
    valid, ticker, error = validate_ticker(msg.text)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    
    await state.update_data(supply_ticker=ticker)
    await msg.answer(f"✅ Залог: <b>{ticker}</b>\n\nВведите тикер займа:")
    await state.set_state(Calc.borrow_ticker)


@dp.message(Calc.borrow_ticker)
async def process_borrow_ticker(msg: types.Message, state: FSMContext):
    valid, ticker, error = validate_ticker(msg.text)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    
    await state.update_data(borrow_ticker=ticker)
    data = await state.get_data()
    await msg.answer(f"✅ Займ: <b>{ticker}</b>\n\nВведите количество {data['supply_ticker']}:")
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
    
    await msg.answer(f"✅ Количество: {value:.6f}\n\n⏳ Получаю цены...")
    
    # Запрос цен параллельно
    cr_task = asyncio.create_task(cryptorank.get_price_usd(ticker))
    cg_task = asyncio.create_task(coingecko.get_price_usd(ticker))
    
    cr_price, cg_price = await asyncio.gather(cr_task, cg_task)
    
    if cr_price is not None or cg_price is not None:
        await state.update_data(cryptorank_price=cr_price, coingecko_price=cg_price)
        
        sources = []
        if cg_price:
            sources.append(f"🦎 CoinGecko: {format_price(cg_price)}")
        if cr_price:
            sources.append(f"✅ CryptoRank: {format_price(cr_price)}")
        
        await msg.answer(
            f"💱 <b>Найдены цены {ticker}:</b>\n" +
            "\n".join(f"• {s}" for s in sources) +
            "\n\n<b>Выберите источник:</b>",
            reply_markup=price_choice_kb(cr_price, cg_price)
        )
        await state.set_state(Calc.choose_price)
    else:
        await msg.answer(f"❌ Цена не найдена\n\nВведите цену {ticker} в USD:")
        await state.set_state(Calc.supply_price_manual)


@dp.callback_query(F.data.startswith("price_"))
async def process_price_choice(cb: types.CallbackQuery, state: FSMContext):
    await cb.answer()
    
    data = await state.get_data()
    choice = cb.data.split("_")[1]
    
    if choice == "manual":
        await cb.message.edit_text(f"✏️ Введите цену {data['supply_ticker']} в USD:")
        await state.set_state(Calc.supply_price_manual)
        return
    
    if choice == "cryptorank":
        price = data.get('cryptorank_price')
        source = "cryptorank"
    else:
        price = data.get('coingecko_price')
        source = "coingecko"
    
    if price is None:
        await cb.message.edit_text("❌ Ошибка. Введите цену вручную:")
        await state.set_state(Calc.supply_price_manual)
        return
    
    await state.update_data(supply_price=price, supply_price_source=source)
    
    collateral = data['supply_amount'] * price
    
    await cb.message.edit_text(
        f"✅ Цена: <b>{format_price(price)}</b>\n"
        f"💰 Залог: <b>{format_currency(collateral)}</b>\n\n"
        "Введите <b>Maximum LTV</b> в %:"
    )
    await state.set_state(Calc.max_ltv)


@dp.message(Calc.supply_price_manual)
async def process_manual_price(msg: types.Message, state: FSMContext):
    valid, price, error = validate_number(msg.text, min_val=0.000001)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    
    await state.update_data(supply_price=price, supply_price_source="manual")
    data = await state.get_data()
    collateral = data['supply_amount'] * price
    
    await msg.answer(
        f"✅ Цена: <b>{format_price(price)}</b>\n"
        f"💰 Залог: <b>{format_currency(collateral)}</b>\n\n"
        "Введите <b>Maximum LTV</b> в %:"
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
    collateral = data['supply_amount'] * data['supply_price']
    max_borrow = collateral * (value / 100)
    
    await msg.answer(
        f"✅ <b>Max LTV: {value}%</b>\n"
        f"💰 Макс. займ: <b>{format_currency(max_borrow)}</b>\n\n"
        "Введите <b>Liquidation Threshold</b> в %:"
    )
    await state.set_state(Calc.lt)


@dp.message(Calc.lt)
async def process_lt(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    
    data = await state.get_data()
    max_ltv = data.get('max_ltv', 0) * 100
    
    if value < max_ltv:
        await msg.answer(f"❌ LT ({value}%) < Max LTV ({max_ltv:.0f}%)")
        return
    
    await state.update_data(lt=value / 100)
    await msg.answer(f"✅ <b>LT: {value}%</b>\n\nВыберите режим:", reply_markup=mode_kb)
    await state.set_state(Calc.mode)


@dp.callback_query(F.data.startswith("mode_"))
async def process_mode(cb: types.CallbackQuery, state: FSMContext):
    await cb.answer()
    
    data = await state.get_data()
    collateral = data['supply_amount'] * data['supply_price']
    max_ltv = data['max_ltv']
    
    await state.update_data(mode=cb.data)
    
    if cb.data == "mode_ltv":
        await cb.message.edit_text(
            f"<b>🔢 Режим: по LTV</b>\n\n"
            f"Залог: {format_currency(collateral)}\n"
            f"Max LTV: {max_ltv * 100:.0f}%\n\n"
            "Введите <b>LTV</b> в %:"
        )
        await state.set_state(Calc.ltv)
    else:
        max_borrow = collateral * max_ltv
        await cb.message.edit_text(
            f"<b>💵 Режим: по сумме</b>\n\n"
            f"Залог: {format_currency(collateral)}\n"
            f"Max займ: <b>{format_currency(max_borrow)}</b>\n\n"
            "Введите сумму займа в USD:"
        )
        await state.set_state(Calc.borrow)


@dp.message(Calc.ltv)
async def process_ltv(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    
    data = await state.get_data()
    max_ltv = data.get('max_ltv', 0) * 100
    
    if value > max_ltv:
        await msg.answer(f"❌ LTV ({value}%) > Max LTV ({max_ltv:.0f}%)")
        return
    
    await state.update_data(ltv=value / 100)
    await calculate_position(msg, state)


@dp.message(Calc.borrow)
async def process_borrow(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0)
    if not valid:
        await msg.answer(f"❌ {error}")
        return
    
    data = await state.get_data()
    collateral = data['supply_amount'] * data['supply_price']
    max_borrow = collateral * data['max_ltv']
    
    if value > max_borrow:
        await msg.answer(f"❌ {format_currency(value)} > макс. {format_currency(max_borrow)}")
        return
    
    await state.update_data(borrow=value)
    await calculate_position(msg, state)


async def calculate_position(msg: types.Message, state: FSMContext):
    try:
        data = await state.get_data()
        
        supply_amt = data['supply_amount']
        price = data['supply_price']
        lt = data['lt']
        max_ltv = data['max_ltv']
        
        collateral = supply_amt * price
        
        if data['mode'] == "mode_ltv":
            borrow = collateral * data['ltv']
            ltv = data['ltv']
        else:
            borrow = data['borrow']
            ltv = borrow / collateral if collateral > 0 else 0
        
        hf = calc_hf(collateral, lt, borrow)
        liq_price = calc_liq_price(borrow, supply_amt, lt)
        buffer = ((price - liq_price) / price) * 100 if price > 0 else 0
        
        scenarios = []
        for drop in [10, 20, 30]:
            new_coll = supply_amt * price * (1 - drop / 100)
            scenarios.append((drop, calc_hf(new_coll, lt, borrow)))
        
        calc = {
            'supply_amt': supply_amt,
            'price': price,
            'collateral': collateral,
            'borrow': borrow,
            'ltv_pct': ltv * 100,
            'max_ltv_pct': max_ltv * 100,
            'lt': lt,
            'hf': hf,
            'liq_price': liq_price,
            'buffer': buffer,
            'max_borrow': collateral * max_ltv,
            'scenarios': scenarios
        }
        
        await msg.answer("⏳ Формирую результаты...")
        await msg.answer(build_result(data, calc))
        await msg.answer("✅ Расчет завершен!\n\n/start - новый расчет")
        
        await state.clear()
        
    except Exception as e:
        logger.error(f"Calc error: {e}")
        await msg.answer(f"❌ Ошибка расчета\n\n/start")
        await state.clear()


@dp.message()
async def fallback(msg: types.Message, state: FSMContext):
    if await state.get_state():
        await msg.answer("⚠️ Следуйте инструкциям или /reset")
    else:
        await msg.answer("👋 /start для начала")


@dp.error()
async def error_handler(update: types.Update, exception: Exception):
    logger.error(f"Update {update.update_id} error: {exception}", exc_info=True)
    return True

# =============================================================================
# STARTUP & MAIN
# =============================================================================

async def on_startup():
    logger.info("=" * 50)
    logger.info("🚀 DeFi Calculator Bot v2.4 Starting")
    
    bot_info = await bot.get_me()
    logger.info(f"✅ Bot: @{bot_info.username}")
    
    if cryptorank.is_available():
        logger.info("✅ CryptoRank v1 API configured")
        # Test request
        test = await cryptorank.get_price_usd("BTC")
        if test:
            logger.info(f"✅ CryptoRank test: BTC=${test:,.2f}")
    else:
        logger.info("ℹ️  CryptoRank API not configured")
    
    logger.info("=" * 50)


async def on_shutdown():
    await cryptorank.close()
    await coingecko.close()
    await bot.session.close()
    logger.info("👋 Bot stopped")


async def main():
    try:
        await on_startup()
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        await on_shutdown()


if __name__ == "__main__":
    asyncio.run(main())
