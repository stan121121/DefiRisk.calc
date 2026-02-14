import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
from collections import deque

from aiogram import Bot, Dispatcher, types, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.strategy import FSMStrategy
import aiohttp

# =============================================================================
# CONFIGURATION
# =============================================================================

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("❌ Не установлен токен бота! Установите BOT_TOKEN в переменных окружения")

COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY", "")

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
storage = MemoryStorage()
dp = Dispatcher(storage=storage, fsm_strategy=FSMStrategy.USER_IN_CHAT)

# =============================================================================
# COINMARKETCAP PRICE FETCHER
# =============================================================================

class CoinMarketCapPriceFetcher:
    """CoinMarketCap API price fetcher с кэшированием"""
    
    BASE_URL = "https://pro-api.coinmarketcap.com/v1"
    
    def __init__(self, api_key: str = "", cache_ttl: int = 300):
        self._api_key = api_key
        self._cache: Dict[str, Tuple[float, datetime]] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl)
        self._session: Optional[aiohttp.ClientSession] = None
        self._stats = {
            "total_requests": 0,
            "success": 0,
            "fail": 0,
            "cache_hits": 0,
            "api_calls": 0
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
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
                self._stats["cache_hits"] += 1
                return price
        return None
    
    def _save_to_cache(self, symbol: str, price: float):
        self._cache[symbol] = (price, datetime.now())
    
    def get_stats(self) -> dict:
        cache_hit_rate = (
            self._stats["cache_hits"] / self._stats["total_requests"] * 100 
            if self._stats["total_requests"] > 0 else 0
        )
        success_rate = (
            self._stats["success"] / self._stats["total_requests"] * 100 
            if self._stats["total_requests"] > 0 else 0
        )
        return {
            **self._stats,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "success_rate": f"{success_rate:.1f}%",
            "cache_size": len(self._cache)
        }
    
    async def get_price_usd(self, symbol: str) -> Optional[float]:
        if not self.is_available():
            return None
        
        self._stats["total_requests"] += 1
        symbol = symbol.upper().strip()
        
        # Проверяем кэш
        cached_price = self._get_from_cache(symbol)
        if cached_price is not None:
            return cached_price
        
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/cryptocurrency/quotes/latest"
            headers = {
                "X-CMC_PRO_API_KEY": self._api_key,
                "Accept": "application/json"
            }
            params = {"symbol": symbol}
            
            self._stats["api_calls"] += 1
            
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status == 429:
                    # Rate limit exceeded
                    return None
                
                if resp.status != 200:
                    self._stats["fail"] += 1
                    return None
                
                data = await resp.json()
                
                if "data" not in data or symbol not in data["data"]:
                    self._stats["fail"] += 1
                    return None
                
                coin_data = data["data"][symbol]
                if "quote" not in coin_data or "USD" not in coin_data["quote"]:
                    self._stats["fail"] += 1
                    return None
                
                price = coin_data["quote"]["USD"]["price"]
                if price is None:
                    self._stats["fail"] += 1
                    return None
                
                price_float = float(price)
                self._stats["success"] += 1
                
                # Сохраняем в кэш
                self._save_to_cache(symbol, price_float)
                
                return price_float
                
        except Exception as e:
            self._stats["fail"] += 1
            return None

# Инициализируем fetcher
cmc_fetcher = CoinMarketCapPriceFetcher(api_key=COINMARKETCAP_API_KEY, cache_ttl=300)

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

def price_choice_kb(cmc_price: Optional[float]):
    """Клавиатура выбора источника цены"""
    buttons = []
    
    if cmc_price is not None:
        if cmc_price >= 1:
            price_str = f"${cmc_price:,.2f}"
        elif cmc_price >= 0.01:
            price_str = f"${cmc_price:.4f}"
        else:
            price_str = f"${cmc_price:.8f}"
        
        buttons.append([InlineKeyboardButton(
            text=f"📊 CoinMarketCap: {price_str}",
            callback_data="price_cmc"
        )])
    
    buttons.append([InlineKeyboardButton(
        text="✏️ Ввести вручную",
        callback_data="price_manual"
    )])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

mode_kb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🔢 По LTV", callback_data="mode_ltv")],
    [InlineKeyboardButton(text="💵 По сумме займа", callback_data="mode_borrow")]
])

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_number(text: str, min_val: float = 0, max_val: Optional[float] = None) -> Tuple[bool, float, str]:
    try:
        text = text.replace(",", ".").strip()
        value = float(text)
        if value <= min_val:
            return False, 0, f"Значение должно быть больше {min_val}"
        if max_val is not None and value > max_val:
            return False, 0, f"Значение должно быть не больше {max_val}"
        return True, value, ""
    except (ValueError, TypeError):
        return False, 0, "Пожалуйста, введите корректное число"

def validate_ticker(text: str, max_length: int = 10) -> Tuple[bool, str, str]:
    ticker = text.upper().strip()
    if len(ticker) > max_length:
        return False, "", f"Тикер слишком длинный (максимум {max_length} символов)"
    if not ticker.isalnum():
        return False, "", "Тикер должен содержать только буквы и цифры"
    return True, ticker, ""

def format_currency(value: float) -> str:
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.2f}"

def format_number(value: float, decimals: int = 2) -> str:
    if value == float('inf'):
        return "∞"
    return f"{value:.{decimals}f}"

def calculate_health_factor(collateral: float, lt: float, borrow: float) -> float:
    if borrow <= 0:
        return float('inf')
    return (collateral * lt) / borrow

def calculate_liquidation_price(borrow: float, supply_amount: float, lt: float) -> float:
    denominator = supply_amount * lt
    if denominator <= 0:
        return 0
    return borrow / denominator

def get_position_status(hf: float) -> Tuple[str, str]:
    if hf <= 1.0:
        return "🔴 ЛИКВИДАЦИЯ", "🔴"
    elif hf < 1.3:
        return "🟡 ВНИМАНИЕ", "🟡"
    elif hf < 2.0:
        return "🟢 БЕЗОПАСНО", "🟢"
    else:
        return "🔵 ОЧЕНЬ БЕЗОПАСНО", "🔵"

def build_result_message(data: dict, calculations: dict) -> str:
    status, emoji = get_position_status(calculations['hf'])
    price_source = data.get('supply_price_source', 'manual')
    
    price = calculations['price']
    if price >= 1:
        price_str = f"${price:,.2f}"
    elif price >= 0.01:
        price_str = f"${price:.4f}"
    else:
        price_str = f"${price:.8f}"
    
    liq_price = calculations['liq_price']
    if liq_price >= 1:
        liq_price_str = f"${liq_price:,.2f}"
    elif liq_price >= 0.01:
        liq_price_str = f"${liq_price:.4f}"
    else:
        liq_price_str = f"${liq_price:.8f}"
    
    source_names = {
        "cmc": "CoinMarketCap",
        "manual": "ручной ввод"
    }
    price_display = f"{price_str} ({source_names.get(price_source, 'API')})"
    
    result = (
        f"<b>{emoji} РАСЧЕТ ПОЗИЦИИ</b>\n"
        f"Статус: <b>{status}</b>\n\n"
        
        f"<b>💎 ЗАЛОГ:</b>\n"
        f"• Актив: <b>{data['supply_ticker']}</b>\n"
        f"• Количество: {calculations['supply_amt']:.6f}\n"
        f"• Цена: {price_display}\n"
        f"• Стоимость: <b>{format_currency(calculations['collateral'])}</b>\n\n"
        
        f"<b>💰 ЗАЙМ:</b>\n"
        f"• Актив: <b>{data['borrow_ticker']}</b>\n"
        f"• Сумма: <b>{format_currency(calculations['borrow'])}</b>\n\n"
        
        f"<b>⚙️ ПАРАМЕТРЫ:</b>\n"
        f"• Maximum LTV: {calculations['max_ltv_percent']}%\n"
        f"• Liquidation Threshold: {calculations['lt']*100:.1f}%\n"
        f"• Current LTV: <b>{calculations['ltv_percent']:.2f}%</b>\n\n"
        
        f"<b>📊 РИСКИ:</b>\n"
        f"• Health Factor: <b>{format_number(calculations['hf'], 2)}</b>\n"
    )
    
    if price_source == "manual":
        result += (
            f"• Цена ликвидации: <b>{liq_price_str}</b>\n"
            f"  <i>(при ручной цене залога {price_str})</i>\n"
        )
    else:
        result += f"• Цена ликвидации: <b>{liq_price_str}</b>\n"
    
    result += (
        f"• Буфер безопасности: <b>{calculations['buffer']:.1f}%</b>\n"
        f"• Макс. возможный займ: {format_currency(calculations['max_borrow'])}\n\n"
        
        f"<b>📉 СЦЕНАРИИ (падение цены):</b>\n"
    )
    
    for drop, scen_hf in calculations['scenarios']:
        new_price = calculations['price'] * (1 - drop / 100)
        if new_price >= 1:
            new_price_str = f"${new_price:,.2f}"
        elif new_price >= 0.01:
            new_price_str = f"${new_price:.4f}"
        else:
            new_price_str = f"${new_price:.8f}"
        result += f"• -{drop}% ({new_price_str}) → HF: {format_number(scen_hf, 2)}\n"
    
    if calculations['hf'] < 1.3:
        result += (
            "\n<b>⚠️ РЕКОМЕНДАЦИИ:</b>\n"
            "• Увеличьте залог для повышения HF\n"
            "• Уменьшите сумму займа\n"
            "• Подготовьте средства для пополнения\n"
            "• Установите алерты на изменение цены"
        )
    
    if price_source == "manual":
        result += (
            f"\n\n💡 <i>Цена {data['supply_ticker']} введена вручную. "
            f"При следующем расчете потребуется ввести заново.</i>"
        )
    
    return result

# =============================================================================
# COMMAND HANDLERS
# =============================================================================

@dp.message(Command("start"))
async def start_cmd(msg: types.Message, state: FSMContext):
    await state.clear()
    
    cmc_status = "✅" if cmc_fetcher.is_available() else "❌"
    
    await msg.answer(
        "🤖 <b>DeFi Risk.calc</b>\n"
        "<i>Калькулятор кредитных позиций в DeFi</i>\n\n"
        
        f"<b>📡 Источники цен:</b>\n"
        f"{cmc_status} CoinMarketCap API\n"
        f"✅ Ручной ввод (любые токены)\n\n"
        
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Введите <b>тикер Supply актива</b>\n"
        "(например: ETH, BTC, SOL)"
    )
    await state.set_state(Calc.supply_ticker)

@dp.message(Command("reset", "cancel"))
async def reset_cmd(msg: types.Message, state: FSMContext):
    await state.clear()
    await msg.answer("✅ Расчет сброшен. Используйте /start для нового расчета")

@dp.message(Command("help"))
async def help_cmd(msg: types.Message):
    await msg.answer(
        "<b>📖 Справка</b>\n\n"
        "<b>Команды:</b>\n"
        "/start - начать расчет\n"
        "/reset - сбросить расчет\n"
        "/stats - статистика API\n\n"
        
        "<b>Порядок ввода:</b>\n"
        "1️⃣ Тикер залога\n"
        "2️⃣ Тикер займа\n"
        "3️⃣ Количество залога\n"
        "4️⃣ Цена (API/ручная)\n"
        "5️⃣ Maximum LTV\n"
        "6️⃣ Liquidation Threshold\n"
        "7️⃣ Режим расчета\n"
        "8️⃣ LTV или сумма займа"
    )

@dp.message(Command("stats"))
async def stats_cmd(msg: types.Message):
    stats = cmc_fetcher.get_stats()
    cmc_status = "настроен ✅" if cmc_fetcher.is_available() else "не настроен ❌"
    
    await msg.answer(
        f"<b>📊 Статистика API</b>\n\n"
        f"<b>CoinMarketCap:</b> {cmc_status}\n"
        f"Всего запросов: {stats['total_requests']}\n"
        f"Успешных: {stats['success']}\n"
        f"Ошибок: {stats['fail']}\n"
        f"Успешность: {stats.get('success_rate', '0%')}\n"
        f"API вызовов: {stats['api_calls']}\n"
        f"Из кэша: {stats['cache_hits']}\n"
        f"Процент кэша: {stats.get('cache_hit_rate', '0%')}\n"
    )

# =============================================================================
# STATE HANDLERS
# =============================================================================

@dp.message(Calc.supply_ticker)
async def process_supply_ticker(msg: types.Message, state: FSMContext):
    valid, ticker, error = validate_ticker(msg.text)
    if not valid:
        await msg.answer(f"❌ {error}\n\nВведите корректный тикер:")
        return
    
    await state.update_data(supply_ticker=ticker)
    
    await msg.answer(
        f"✅ <b>Supply актив:</b> {ticker}\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Введите <b>тикер Borrow актива</b>\n"
        "(например: USDC, USDT, PYUSD)"
    )
    await state.set_state(Calc.borrow_ticker)

@dp.message(Calc.borrow_ticker)
async def process_borrow_ticker(msg: types.Message, state: FSMContext):
    valid, ticker, error = validate_ticker(msg.text)
    if not valid:
        await msg.answer(f"❌ {error}\n\nВведите корректный тикер:")
        return
    
    await state.update_data(borrow_ticker=ticker)
    data = await state.get_data()
    
    await msg.answer(
        f"✅ <b>Borrow актив:</b> {ticker}\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"Введите <b>количество {data['supply_ticker']}</b>"
    )
    await state.set_state(Calc.supply_amount)

@dp.message(Calc.supply_amount)
async def process_supply_amount(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0.000001)
    if not valid:
        await msg.answer(f"❌ {error}\n\nВведите количество:")
        return
    
    await state.update_data(supply_amount=value)
    data = await state.get_data()
    ticker = data['supply_ticker']
    
    await msg.answer(f"✅ Количество: {value:.6f}")
    
    # Пытаемся получить цену из CoinMarketCap
    if cmc_fetcher.is_available():
        await msg.answer(f"⏳ Получаю цену {ticker} из CoinMarketCap...")
        cmc_price = await cmc_fetcher.get_price_usd(ticker)
        
        if cmc_price is not None:
            await state.update_data(cmc_price=cmc_price)
            
            if cmc_price >= 1:
                price_str = f"${cmc_price:,.2f}"
            elif cmc_price >= 0.01:
                price_str = f"${cmc_price:.4f}"
            else:
                price_str = f"${cmc_price:.8f}"
            
            await msg.answer(
                f"💱 <b>Найдена цена {ticker}:</b>\n"
                f"• 📊 CoinMarketCap: {price_str}\n\n"
                f"<b>Выберите источник:</b>",
                reply_markup=price_choice_kb(cmc_price)
            )
            await state.set_state(Calc.choose_price)
            return
    
    # Если CoinMarketCap не доступен или не нашел цену
    await msg.answer(
        f"❌ Цена {ticker} не найдена в API или API не настроен\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"Введите <b>цену {ticker}</b> в USD вручную:"
    )
    await state.set_state(Calc.supply_price_manual)

@dp.callback_query(F.data.startswith("price_"))
async def process_price_choice(cb: types.CallbackQuery, state: FSMContext):
    await cb.answer()
    
    data = await state.get_data()
    choice = cb.data.split("_")[1]
    
    if choice == "manual":
        await cb.message.edit_text(
            f"✏️ Введите цену <b>{data['supply_ticker']}</b> в USD вручную:"
        )
        await state.set_state(Calc.supply_price_manual)
        return
    
    # Используем CoinMarketCap цену
    if choice == "cmc":
        price = data.get('cmc_price')
        source = "cmc"
        source_name = "CoinMarketCap"
    
    if price is None:
        await cb.message.edit_text("❌ Ошибка получения цены. Введите вручную:")
        await state.set_state(Calc.supply_price_manual)
        return
    
    await state.update_data(supply_price=price, supply_price_source=source)
    
    supply_amount = data['supply_amount']
    collateral_value = supply_amount * price
    
    if price >= 1:
        price_str = f"${price:,.2f}"
    elif price >= 0.01:
        price_str = f"${price:.4f}"
    elif price >= 0.0001:
        price_str = f"${price:.6f}"
    else:
        price_str = f"${price:.8f}"
    
    await cb.message.edit_text(
        f"✅ Цена ({source_name}): <b>{price_str}</b>\n"
        f"💰 Стоимость залога: <b>{format_currency(collateral_value)}</b>\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Введите <b>Maximum LTV</b> в %\n"
        "(например: 65)"
    )
    await state.set_state(Calc.max_ltv)

@dp.message(Calc.supply_price_manual)
async def process_supply_price_manual(msg: types.Message, state: FSMContext):
    valid, price, error = validate_number(msg.text, min_val=0.000001)
    if not valid:
        await msg.answer(f"❌ {error}\n\nВведите цену:")
        return
    
    data = await state.get_data()
    ticker = data['supply_ticker']
    amount = data['supply_amount']
    
    await state.update_data(supply_price=price, supply_price_source="manual")
    collateral_value = amount * price
    
    if price >= 1:
        price_str = f"${price:,.2f}"
    elif price >= 0.01:
        price_str = f"${price:.4f}"
    elif price >= 0.0001:
        price_str = f"${price:.6f}"
    else:
        price_str = f"${price:.8f}"
    
    await msg.answer(
        f"✅ Цена (ручной ввод): <b>{price_str}</b>\n"
        f"💰 Стоимость залога: <b>{format_currency(collateral_value)}</b>\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Введите <b>Maximum LTV</b> в %\n"
        "(например: 65)"
    )
    await state.set_state(Calc.max_ltv)

@dp.message(Calc.max_ltv)
async def process_max_ltv(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    if not valid:
        await msg.answer(f"❌ {error}\n\nMax LTV должен быть 0-100%. Введите:")
        return
    
    await state.update_data(max_ltv=value / 100)
    
    data = await state.get_data()
    supply_amount = data.get('supply_amount', 0)
    supply_price = data.get('supply_price', 0)
    collateral_value = supply_amount * supply_price
    max_possible_borrow = collateral_value * (value / 100)
    
    await msg.answer(
        f"✅ <b>Maximum LTV: {value}%</b>\n"
        f"💰 Макс. возможный займ: <b>{format_currency(max_possible_borrow)}</b>\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Введите <b>Liquidation Threshold (LT)</b> в %\n"
        "(например: 75)"
    )
    await state.set_state(Calc.lt)

@dp.message(Calc.lt)
async def process_lt(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    if not valid:
        await msg.answer(f"❌ {error}\n\nLT должен быть 0-100%. Введите:")
        return
    
    data = await state.get_data()
    max_ltv = data.get('max_ltv', 0) * 100
    
    if value < max_ltv:
        await msg.answer(
            f"❌ <b>Ошибка:</b> Liquidation Threshold ({value}%) должен быть "
            f"больше или равен Maximum LTV ({max_ltv:.0f}%)\n\n"
            "Введите корректное значение LT:"
        )
        return
    
    await state.update_data(lt=value / 100)
    
    await msg.answer(
        f"✅ <b>Liquidation Threshold: {value}%</b>\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Выберите <b>режим расчета</b>:",
        reply_markup=mode_kb
    )
    await state.set_state(Calc.mode)

@dp.callback_query(F.data.startswith("mode_"))
async def process_mode(cb: types.CallbackQuery, state: FSMContext):
    await cb.answer()
    mode = cb.data
    data = await state.get_data()
    
    supply_amount = data.get('supply_amount', 0)
    supply_price = data.get('supply_price', 0)
    collateral_value = supply_amount * supply_price
    max_ltv = data.get('max_ltv', 0)
    
    await state.update_data(mode=mode)
    
    if mode == "mode_ltv":
        await cb.message.edit_text(
            f"<b>🔢 Режим: Расчет по LTV</b>\n\n"
            f"Стоимость залога: {format_currency(collateral_value)}\n"
            f"Maximum LTV: {max_ltv * 100:.0f}%\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Введите <b>LTV</b> в %\n"
            "(например: 50)"
        )
        await state.set_state(Calc.ltv)
    else:
        max_possible_borrow = collateral_value * max_ltv
        
        await cb.message.edit_text(
            f"<b>💵 Режим: Расчет по сумме займа</b>\n\n"
            f"Стоимость залога: {format_currency(collateral_value)}\n"
            f"Maximum LTV: {max_ltv * 100:.0f}%\n"
            f"<b>Макс. возможный займ: {format_currency(max_possible_borrow)}</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Введите <b>сумму займа</b> в USD\n"
            f"(максимум: {format_currency(max_possible_borrow)})"
        )
        await state.set_state(Calc.borrow)

@dp.message(Calc.ltv)
async def process_ltv(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    if not valid:
        await msg.answer(f"❌ {error}\n\nLTV должен быть 0-100%. Введите:")
        return
    
    data = await state.get_data()
    max_ltv = data.get('max_ltv', 0) * 100
    
    if value > max_ltv:
        await msg.answer(
            f"❌ <b>Ошибка:</b> LTV ({value}%) не может превышать "
            f"Maximum LTV ({max_ltv:.0f}%)\n\n"
            "Введите корректное значение:"
        )
        return
    
    await state.update_data(ltv=value / 100)
    await calculate_position(msg, state)

@dp.message(Calc.borrow)
async def process_borrow(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0)
    if not valid:
        await msg.answer(f"❌ {error}\n\nВведите сумму:")
        return
    
    data = await state.get_data()
    supply_amount = data.get('supply_amount', 0)
    supply_price = data.get('supply_price', 0)
    collateral_value = supply_amount * supply_price
    max_ltv = data.get('max_ltv', 0)
    max_borrow_allowed = collateral_value * max_ltv
    
    if value > max_borrow_allowed:
        await msg.answer(
            f"❌ <b>Ошибка:</b> Сумма займа ({format_currency(value)}) превышает "
            f"максимально возможный займ ({format_currency(max_borrow_allowed)}) "
            f"при Max LTV {max_ltv*100:.0f}%\n\n"
            "Введите корректную сумму:"
        )
        return
    
    await state.update_data(borrow=value)
    await calculate_position(msg, state)

# =============================================================================
# CALCULATION
# =============================================================================

async def calculate_position(msg: types.Message, state: FSMContext):
    try:
        data = await state.get_data()
        
        required = ['supply_ticker', 'borrow_ticker', 'supply_amount', 
                   'supply_price', 'lt', 'max_ltv', 'mode']
        if not all(f in data for f in required):
            await msg.answer("❌ Недостаточно данных. Начните заново с /start")
            await state.clear()
            return
        
        supply_amt = data['supply_amount']
        price = data['supply_price']
        lt = data['lt']
        max_ltv = data['max_ltv']
        mode = data['mode']
        
        collateral = supply_amt * price
        
        if mode == "mode_ltv":
            ltv = data.get('ltv')
            if ltv is None:
                await msg.answer("❌ Отсутствует LTV")
                await state.clear()
                return
            borrow = collateral * ltv
        else:
            borrow = data.get('borrow')
            if borrow is None:
                await msg.answer("❌ Отсутствует сумма займа")
                await state.clear()
                return
            ltv = borrow / collateral if collateral > 0 else 0
        
        ltv_percent = ltv * 100
        
        hf = calculate_health_factor(collateral, lt, borrow)
        liq_price = calculate_liquidation_price(borrow, supply_amt, lt)
        max_borrow = collateral * max_ltv
        buffer = ((price - liq_price) / price) * 100 if price > 0 else 0
        
        scenarios = []
        for drop in [10, 20, 30]:
            new_price = price * (1 - drop / 100)
            new_coll = supply_amt * new_price
            scen_hf = calculate_health_factor(new_coll, lt, borrow)
            scenarios.append((drop, scen_hf))
        
        calculations = {
            'supply_amt': supply_amt,
            'price': price,
            'collateral': collateral,
            'borrow': borrow,
            'ltv_percent': ltv_percent,
            'max_ltv_percent': max_ltv * 100,
            'lt': lt,
            'hf': hf,
            'liq_price': liq_price,
            'buffer': buffer,
            'max_borrow': max_borrow,
            'scenarios': scenarios
        }
        
        result_message = build_result_message(data, calculations)
        
        await msg.answer("⏳ Формирую результаты...")
        await msg.answer(result_message)
        await msg.answer(
            "━━━━━━━━━━━━━━━━━━━━\n"
            "✅ Расчет завершен!\n\n"
            "/start - новый расчет"
        )
        
        await state.clear()
        
    except Exception as e:
        await msg.answer(f"❌ Ошибка: {str(e)}\n\nИспользуйте /start")
        await state.clear()

# =============================================================================
# FALLBACK & ERROR HANDLERS
# =============================================================================

@dp.message()
async def fallback_handler(msg: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state:
        await msg.answer("⚠️ Следуйте инструкциям или используйте /reset")
    else:
        await msg.answer("👋 Привет! Используйте /start для начала расчета")

@dp.error()
async def error_handler(event, exception):
    print(f"❌ Ошибка: {exception}")
    return True

# =============================================================================
# STARTUP & SHUTDOWN
# =============================================================================

async def on_startup():
    print("\n" + "=" * 60)
    print("🚀 DeFi Position Calculator Bot")
    print("=" * 60)
    
    bot_info = await bot.get_me()
    print(f"✅ Бот: @{bot_info.username}")
    
    # Удаляем вебхук для чистого запуска
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        print("✅ Удален вебхук и очищены ожидающие обновления")
    except Exception as e:
        print(f"⚠️ Не удалось удалить вебхук: {e}")
    
    if cmc_fetcher.is_available():
        print("✅ CoinMarketCap API: настроен")
    else:
        print("ℹ️  CoinMarketCap API: не настроен (используйте ручной ввод цен)")
    
    print("=" * 60)
    print("✅ БОТ ГОТОВ")
    print("=" * 60 + "\n")

async def on_shutdown():
    await cmc_fetcher.close()
    await bot.session.close()
    print("\n👋 Бот остановлен")

# =============================================================================
# MAIN
# =============================================================================

async def main():
    try:
        await on_startup()
        
        # Настройки polling для предотвращения конфликтов
        polling_config = {
            "allowed_updates": dp.resolve_used_update_types(),
            "close_timeout": 10,
            "timeout": 30
        }
        
        await dp.start_polling(bot, **polling_config)
    except KeyboardInterrupt:
        print("\n⚠️ Остановка...")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await on_shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 До свидания!")
