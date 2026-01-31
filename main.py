"""
DeFi Position Calculator Bot с автоматическим получением цен
Интеграция CoinGecko Price Fetcher + DeFi калькулятор
"""

import asyncio
import os
from aiogram import Bot, Dispatcher, types, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.strategy import FSMStrategy
from typing import Tuple, Optional
from dataclasses import dataclass

# Импорт price fetcher
import aiohttp
from datetime import datetime, timedelta
from collections import deque

# ---------- PRICE FETCHER ----------
class CoinGeckoPriceFetcher:
    """Price fetcher с кэшированием и rate limiting"""
    
    COINGECKO_IDS = {
        "ETH": "ethereum",
        "BTC": "bitcoin",
        "SOL": "solana",
        "USDC": "usd-coin",
        "USDT": "tether",
        "DAI": "dai",
        "BNB": "binancecoin",
        "ADA": "cardano",
        "AVAX": "avalanche-2",
        "MATIC": "matic-network",
    }
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, cache_ttl: int = 300, max_requests_per_minute: int = 5):
        self._cache = {}
        self._cache_ttl = timedelta(seconds=cache_ttl)
        self._session = None
        self._max_requests_per_minute = max_requests_per_minute
        self._request_times = deque(maxlen=max_requests_per_minute)
        self._rate_limit_lock = asyncio.Lock()
        self._stats = {"total_requests": 0, "cache_hits": 0, "api_calls": 0}
    
    async def _get_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=15)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _wait_for_rate_limit(self):
        async with self._rate_limit_lock:
            now = datetime.now()
            while self._request_times and (now - self._request_times[0]).total_seconds() > 60:
                self._request_times.popleft()
            
            if len(self._request_times) >= self._max_requests_per_minute:
                oldest_request = self._request_times[0]
                wait_time = 60 - (now - oldest_request).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time + 0.5)
            
            self._request_times.append(now)
    
    def _get_from_cache(self, symbol: str):
        if symbol in self._cache:
            price, timestamp = self._cache[symbol]
            if datetime.now() - timestamp < self._cache_ttl:
                self._stats["cache_hits"] += 1
                return price
        return None
    
    def _save_to_cache(self, symbol: str, price: float):
        self._cache[symbol] = (price, datetime.now())
    
    async def get_price_usd(self, symbol: str, use_cache: bool = True):
        symbol = symbol.upper().strip()
        self._stats["total_requests"] += 1
        
        if use_cache:
            cached_price = self._get_from_cache(symbol)
            if cached_price is not None:
                return cached_price
        
        if symbol not in self.COINGECKO_IDS:
            return None
        
        url = f"{self.BASE_URL}/simple/price"
        params = {"ids": self.COINGECKO_IDS[symbol], "vs_currencies": "usd"}
        
        try:
            await self._wait_for_rate_limit()
            session = await self._get_session()
            self._stats["api_calls"] += 1
            
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', '60'))
                    await asyncio.sleep(retry_after)
                    return await self.get_price_usd(symbol, use_cache=False)
                
                response.raise_for_status()
                data = await response.json()
                
                coin_id = self.COINGECKO_IDS[symbol]
                price = data[coin_id]["usd"]
                
                if use_cache:
                    self._save_to_cache(symbol, price)
                
                return price
        except Exception as e:
            print(f"❌ Ошибка получения цены {symbol}: {e}")
            return None
    
    @classmethod
    def is_supported(cls, symbol: str) -> bool:
        return symbol.upper().strip() in cls.COINGECKO_IDS
    
    @classmethod
    def get_supported_symbols(cls):
        return list(cls.COINGECKO_IDS.keys())

# ---------- CONFIGURATION ----------
TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    raise ValueError("Не установлен токен бота. Установите переменную окружения BOT_TOKEN")

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher(storage=MemoryStorage(), fsm_strategy=FSMStrategy.USER_IN_CHAT)

# Глобальный price fetcher
price_fetcher = CoinGeckoPriceFetcher(cache_ttl=300, max_requests_per_minute=5)

# ---------- STATES ----------
class Calc(StatesGroup):
    supply_ticker = State()
    borrow_ticker = State()
    supply_amount = State()
    # supply_price убрали - получаем автоматически!
    mode = State()
    ltv = State()
    borrow = State()
    lt = State()
    max_ltv = State()

# ---------- KEYBOARDS ----------
mode_kb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🔢 По LTV", callback_data="mode_ltv")],
    [InlineKeyboardButton(text="💵 По сумме займа", callback_data="mode_borrow")]
])

# ---------- HELPERS ----------
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
        return False, "", f"Тикер слишком длинный (max {max_length} символов)"
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

# ---------- COMMANDS ----------
@dp.message(Command("start"))
async def start_cmd(msg: types.Message, state: FSMContext):
    await state.clear()
    supported = price_fetcher.get_supported_symbols()
    await msg.answer(
        "<b>📊 DeFi Risk calculator</b>\n"
        "<i>с автоматическим получением цен</i>\n\n"
        f"<b>Поддерживаемые монеты ({len(supported)}):</b>\n"
        f"{', '.join(supported[:8])}...\n\n"
        "Введите тикер залогового актива (например: ETH, SOL, BTC):"
    )
    await state.set_state(Calc.supply_ticker)

@dp.message(Command("reset"))
async def reset_cmd(msg: types.Message, state: FSMContext):
    await state.clear()
    await msg.answer("✅ Состояние сброшено. Используйте /start для начала.")

@dp.message(Command("help"))
async def help_cmd(msg: types.Message):
    await msg.answer(
        "<b>📖 Помощь по боту</b>\n\n"
        "<b>Команды:</b>\n"
        "• /start - начать расчет\n"
        "• /reset - сбросить расчет\n"
        "• /supported - список монет\n"
        "• /help - помощь\n\n"
        "<b>Расчитывает:</b>\n"
        "• Health Factor\n"
        "• Цену ликвидации\n"
        "• Макс. займ\n"
        "• Сценарии падения цены\n\n"
        "💡 Цены получаются автоматически!"
    )

@dp.message(Command("supported"))
async def supported_cmd(msg: types.Message):
    supported = price_fetcher.get_supported_symbols()
    cols = 4
    rows = []
    for i in range(0, len(supported), cols):
        row = " | ".join(f"<code>{coin}</code>" for coin in supported[i:i+cols])
        rows.append(row)
    
    await msg.answer(
        f"<b>💎 Поддерживаемые монеты ({len(supported)})</b>\n\n"
        + "\n".join(rows)
    )

# ---------- FLOW ----------
@dp.message(Calc.supply_ticker)
async def process_supply_ticker(msg: types.Message, state: FSMContext):
    valid, ticker, error = validate_ticker(msg.text)
    
    if not valid:
        await msg.answer(f"❌ {error}\n\nВведите корректный тикер:")
        return
    
    # Проверяем поддержку
    if not price_fetcher.is_supported(ticker):
        await msg.answer(
            f"❌ Тикер <b>{ticker}</b> не поддерживается\n\n"
            f"Используйте /supported для списка монет"
        )
        return
    
    await state.update_data(supply_ticker=ticker)
    await msg.answer(
        f"✅ Залоговый актив: <b>{ticker}</b>\n\n"
        "Введите тикер заимствуемого актива:"
    )
    await state.set_state(Calc.borrow_ticker)

@dp.message(Calc.borrow_ticker)
async def process_borrow_ticker(msg: types.Message, state: FSMContext):
    valid, ticker, error = validate_ticker(msg.text)
    
    if not valid:
        await msg.answer(f"❌ {error}\n\nВведите корректный тикер:")
        return
    
    if not price_fetcher.is_supported(ticker):
        await msg.answer(
            f"❌ Тикер <b>{ticker}</b> не поддерживается\n\n"
            f"Используйте /supported для списка монет"
        )
        return
    
    await state.update_data(borrow_ticker=ticker)
    await msg.answer(
        f"✅ Заимствуемый актив: <b>{ticker}</b>\n\n"
        "Введите количество залогового актива:"
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
    ticker = data.get('supply_ticker')
    
    # Автоматически получаем цену!
    await msg.answer(
        f"✅ Залоговый актив: <b>{ticker}</b>\n"
        f"✅ Количество: <b>{value:.6f}</b>\n\n"
        f"⏳ Получаю текущую цену {ticker}..."
    )
    
    price = await price_fetcher.get_price_usd(ticker)
    
    if price is None:
        await msg.answer(
            f"❌ Не удалось получить цену {ticker}\n\n"
            "Попробуйте позже или начните заново (/start)"
        )
        await state.clear()
        return
    
    await state.update_data(supply_price=price)
    collateral_value = value * price
    
    await msg.answer(
        f"<b>📊 Предварительный расчет</b>\n\n"
        f"Залоговый актив: <b>{ticker}</b>\n"
        f"Количество: {value:.6f}\n"
        f"Цена (CoinGecko): <b>${price:,.2f}</b>\n"
        f"<b>💰 Стоимость залога: {format_currency(collateral_value)}</b>\n\n"
        "Выберите режим расчета:",
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
    
    await state.update_data(mode=mode)
    
    if mode == "mode_ltv":
        await cb.message.edit_text(
            f"<b>🔢 Режим: Расчет по LTV</b>\n\n"
            f"Стоимость залога: {format_currency(collateral_value)}\n\n"
            "Введите Loan-to-Value (LTV) в % (например: 50):"
        )
        await state.set_state(Calc.ltv)
    else:
        await cb.message.edit_text(
            f"<b>💵 Режим: Расчет по сумме займа</b>\n\n"
            f"Стоимость залога: {format_currency(collateral_value)}\n\n"
            "Введите сумму займа в USD:"
        )
        await state.set_state(Calc.borrow)

@dp.message(Calc.ltv)
async def process_ltv(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    
    if not valid:
        await msg.answer(f"❌ {error}\n\nLTV должен быть от 0 до 100%. Введите LTV:")
        return
    
    await state.update_data(ltv=value / 100)
    data = await state.get_data()
    
    supply_amount = data.get('supply_amount', 0)
    supply_price = data.get('supply_price', 0)
    collateral_value = supply_amount * supply_price
    borrow_amount = collateral_value * (value / 100)
    
    await msg.answer(
        f"✅ <b>LTV: {value}%</b>\n"
        f"Сумма займа: {format_currency(borrow_amount)}\n\n"
        "Введите Liquidation Threshold (LT) в % (например: 75):"
    )
    await state.set_state(Calc.lt)

@dp.message(Calc.borrow)
async def process_borrow(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0)
    
    if not valid:
        await msg.answer(f"❌ {error}\n\nВведите сумму займа:")
        return
    
    data = await state.get_data()
    supply_amount = data.get('supply_amount', 0)
    supply_price = data.get('supply_price', 0)
    collateral_value = supply_amount * supply_price
    
    if value > collateral_value:
        await msg.answer(
            f"❌ Сумма займа ({format_currency(value)}) превышает "
            f"стоимость залога ({format_currency(collateral_value)})\n\n"
            "Введите корректную сумму:"
        )
        return
    
    await state.update_data(borrow=value)
    ltv_percent = (value / collateral_value) * 100 if collateral_value > 0 else 0
    
    await msg.answer(
        f"✅ <b>Сумма займа: {format_currency(value)}</b>\n"
        f"LTV: {ltv_percent:.1f}%\n\n"
        "Введите Liquidation Threshold (LT) в %:"
    )
    await state.set_state(Calc.lt)

@dp.message(Calc.lt)
async def process_lt(msg: types.Message, state: FSMContext):
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    
    if not valid:
        await msg.answer(f"❌ {error}\n\nLT должен быть от 0 до 100%. Введите LT:")
        return
    
    await state.update_data(lt=value / 100)
    await msg.answer(
        f"✅ <b>Liquidation Threshold: {value}%</b>\n\n"
        "Введите Maximum LTV в % (например: 65):"
    )
    await state.set_state(Calc.max_ltv)

@dp.message(Calc.max_ltv)
async def calculate_position(msg: types.Message, state: FSMContext):
    try:
        valid, max_ltv_input, error = validate_number(msg.text, min_val=0, max_val=100)
        if not valid:
            await msg.answer(f"❌ {error}\n\nВведите Maximum LTV:")
            return
        
        max_ltv = max_ltv_input / 100
        data = await state.get_data()
        
        # Проверяем данные
        required = ['supply_ticker', 'borrow_ticker', 'supply_amount', 'supply_price', 'lt', 'mode']
        if not all(field in data for field in required):
            await msg.answer("❌ Недостаточно данных. Начните заново с /start")
            await state.clear()
            return
        
        supply_amt = data['supply_amount']
        price = data['supply_price']
        lt = data['lt']
        mode = data['mode']
        
        collateral = supply_amt * price
        
        # Рассчитываем займ и LTV
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
        
        # Валидация
        if ltv > max_ltv:
            await msg.answer(
                f"❌ LTV ({ltv_percent:.1f}%) превышает Max LTV ({max_ltv_input}%)"
            )
            return
        
        if lt <= ltv:
            await msg.answer(
                f"❌ LT ({lt*100:.1f}%) должен быть больше LTV ({ltv_percent:.1f}%)"
            )
            return
        
        # Расчеты
        hf = calculate_health_factor(collateral, lt, borrow)
        liq_price = calculate_liquidation_price(borrow, supply_amt, lt)
        max_borrow = collateral * max_ltv
        buffer = ((price - liq_price) / price) * 100 if price > 0 else 0
        
        # Сценарии
        scenarios = []
        for drop in [10, 20, 30]:
            new_price = price * (1 - drop / 100)
            new_coll = supply_amt * new_price
            scen_hf = calculate_health_factor(new_coll, lt, borrow)
            scenarios.append((drop, scen_hf))
        
        status, emoji = get_position_status(hf)
        
        # Результат
        result = (
            f"<b>{emoji} РАСЧЕТ ПОЗИЦИИ</b>\n"
            f"Статус: <b>{status}</b>\n\n"
            
            f"<b>💎 ЗАЛОГ:</b>\n"
            f"• {data['supply_ticker']}: {supply_amt:.6f}\n"
            f"• Цена: ${price:,.2f}\n"
            f"• Стоимость: <b>{format_currency(collateral)}</b>\n\n"
            
            f"<b>💰 ЗАЙМ:</b>\n"
            f"• {data['borrow_ticker']}: <b>{format_currency(borrow)}</b>\n\n"
            
            f"<b>⚙️ ПАРАМЕТРЫ:</b>\n"
            f"• Current LTV: <b>{ltv_percent:.2f}%</b>\n"
            f"• Maximum LTV: {max_ltv_input}%\n"
            f"• Liquidation Threshold: {lt*100:.1f}%\n\n"
            
            f"<b>📊 РИСКИ:</b>\n"
            f"• Health Factor: <b>{format_number(hf, 2)}</b>\n"
            f"• Цена ликвидации: <b>${liq_price:.2f}</b>\n"
            f"• Буфер: <b>{buffer:.1f}%</b>\n"
            f"• Макс. займ: {format_currency(max_borrow)}\n\n"
            
            f"<b>📉 СЦЕНАРИИ:</b>\n"
        )
        
        for drop, scen_hf in scenarios:
            result += f"• -{drop}% (${price*(1-drop/100):.2f}) → HF: {format_number(scen_hf, 2)}\n"
        
        if hf < 1.3:
            result += (
                "\n<b>⚠️ РЕКОМЕНДАЦИИ:</b>\n"
                "• Увеличьте залог\n"
                "• Уменьшите займ\n"
                "• Установите алерты"
            )
        
        await msg.answer(result)
        await msg.answer("📝 Для нового расчета: /start")
        await state.clear()
        
    except Exception as e:
        await msg.answer(f"❌ Ошибка: {str(e)}\n\nИспользуйте /start")
        await state.clear()

@dp.message()
async def fallback_handler(msg: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state:
        await msg.answer("⚠️ Следуйте инструкциям или используйте /reset")
    else:
        await msg.answer(
            "👋 Привет! Я помогу рассчитать DeFi позицию.\n\n"
            "/start - начать\n/help - помощь"
        )

@dp.error()
async def error_handler(event, exception):
    print(f"❌ Ошибка: {exception}")
    return True

# ---------- STARTUP/SHUTDOWN ----------
async def on_startup():
    print("=" * 60)
    print("🚀 DeFi Calculator Bot с Auto Price Fetching")
    print("=" * 60)
    bot_info = await bot.get_me()
    print(f"✅ Бот: @{bot_info.username}")
    
    # Тест CoinGecko
    price = await price_fetcher.get_price_usd("BTC")
    if price:
        print(f"✅ CoinGecko работает (BTC: ${price:,.2f})")
    else:
        print("⚠️ CoinGecko может быть недоступен")
    
    print("=" * 60)
    print("✅ БОТ ГОТОВ")
    print("=" * 60 + "\n")

async def main():
    try:
        await on_startup()
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except KeyboardInterrupt:
        print("\n⚠️ Остановка...")
    finally:
        await price_fetcher.close()
        await bot.session.close()
        print("👋 Бот остановлен")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 До свидания!")

