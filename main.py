"""
=============================================================================
DeFi Position Calculator Bot - Финальная версия v2.3
=============================================================================

Изменения v2.3:
✅ Добавлена отладка для CryptoRank API
✅ Улучшена обработка ошибок CryptoRank
✅ Логирование всех API запросов
✅ Автоматический fallback на CoinGecko при ошибках CryptoRank

=============================================================================
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
from typing import Tuple, Optional, Dict
import aiohttp
import json
from datetime import datetime, timedelta
from collections import deque

# =============================================================================
# PRICE FETCHERS - УЛУЧШЕННЫЕ С ОТЛАДКОЙ
# =============================================================================

class CryptoRankPriceFetcher:
    """CryptoRank API price fetcher с расширенной отладкой"""
    
    BASE_URL = "https://api.cryptorank.io/v2/currencies"
    
    def __init__(self, api_key: str = ""):
        self._api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._stats = {"total": 0, "success": 0, "fail": 0, "errors": []}
    
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
        available = bool(self._api_key)
        print(f"🔍 CryptoRank доступен: {available}, ключ: {'есть' if self._api_key else 'нет'}")
        return available
    
    async def get_price_usd(self, symbol: str) -> Optional[float]:
        if not self.is_available():
            print(f"❌ CryptoRank не доступен для {symbol}")
            return None
        
        self._stats["total"] += 1
        symbol = symbol.upper().strip()
        print(f"🔍 Запрос CryptoRank для {symbol}...")
        
        try:
            session = await self._get_session()
            headers = {"X-Api-Key": self._api_key}
            params = {"symbols": symbol}
            
            print(f"🔍 Запрос к CryptoRank: {self.BASE_URL}")
            print(f"🔍 Заголовки: { {k: '***' if 'Key' in k else v for k, v in headers.items()} }")
            print(f"🔍 Параметры: {params}")
            
            async with session.get(
                self.BASE_URL,
                headers=headers,
                params=params
            ) as resp:
                print(f"🔍 CryptoRank статус: {resp.status}")
                
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"❌ CryptoRank ошибка {resp.status}: {error_text[:200]}")
                    self._stats["fail"] += 1
                    self._stats["errors"].append(f"HTTP {resp.status}: {error_text[:100]}")
                    return None
                
                data = await resp.json()
                print(f"🔍 CryptoRank ответ: {json.dumps(data, indent=2)[:500]}...")
                
                items = data.get("data", [])
                
                if not items:
                    print(f"❌ CryptoRank: нет данных для {symbol}")
                    self._stats["fail"] += 1
                    self._stats["errors"].append(f"No data for {symbol}")
                    return None
                
                try:
                    price = float(items[0]["values"]["USD"]["price"])
                    print(f"✅ CryptoRank цена для {symbol}: ${price}")
                    self._stats["success"] += 1
                    return price
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    print(f"❌ CryptoRank: ошибка парсинга для {symbol}: {e}")
                    print(f"🔍 Структура данных: {items[0].keys() if items else 'нет items'}")
                    if items and 'values' in items[0]:
                        print(f"🔍 Доступные валюты: {list(items[0]['values'].keys())}")
                    self._stats["fail"] += 1
                    self._stats["errors"].append(f"Parse error for {symbol}: {e}")
                    return None
        except aiohttp.ClientError as e:
            print(f"❌ CryptoRank сетевой ошибка для {symbol}: {e}")
            self._stats["fail"] += 1
            self._stats["errors"].append(f"Network error: {str(e)}")
            return None
        except Exception as e:
            print(f"❌ CryptoRank неожиданная ошибка для {symbol}: {e}")
            import traceback
            traceback.print_exc()
            self._stats["fail"] += 1
            self._stats["errors"].append(f"Unexpected error: {str(e)}")
            return None
    
    def get_stats(self) -> dict:
        return {
            **self._stats,
            "success_rate": f"{(self._stats['success'] / self._stats['total'] * 100):.1f}%" if self._stats['total'] > 0 else "0%",
            "recent_errors": self._stats["errors"][-5:] if self._stats["errors"] else []
        }


class CoinGeckoPriceFetcher:
    """Price fetcher с кэшированием и rate limiting"""
    
    COINGECKO_IDS = {
        "ETH": "ethereum",
        "BTC": "bitcoin",
        "SOL": "solana",
        "USDC": "usd-coin",
        "USDT": "tether",
        "DAI": "dai",
        "BUSD": "binance-usd",
        "BNB": "binancecoin",
        "ADA": "cardano",
        "DOT": "polkadot",
        "AVAX": "avalanche-2",
        "MATIC": "matic-network",
        "LINK": "chainlink",
        "UNI": "uniswap",
        "ATOM": "cosmos",
        "XRP": "ripple",
        "LTC": "litecoin",
        "DOGE": "dogecoin",
        "SHIB": "shiba-inu",
        "AAVE": "aave",
    }
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, cache_ttl: int = 300, max_requests_per_minute: int = 5):
        self._cache: Dict[str, Tuple[float, datetime]] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl)
        self._session: Optional[aiohttp.ClientSession] = None
        self._max_requests_per_minute = max_requests_per_minute
        self._request_times = deque(maxlen=max_requests_per_minute)
        self._rate_limit_lock = asyncio.Lock()
        self._stats = {"total_requests": 0, "cache_hits": 0, "api_calls": 0, "errors": []}
    
    async def _get_session(self) -> aiohttp.ClientSession:
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
                    print(f"⏳ CoinGecko rate limit, жду {wait_time:.1f} секунд")
                    await asyncio.sleep(wait_time + 0.5)
            self._request_times.append(now)
    
    def _get_from_cache(self, symbol: str) -> Optional[float]:
        if symbol in self._cache:
            price, timestamp = self._cache[symbol]
            if datetime.now() - timestamp < self._cache_ttl:
                self._stats["cache_hits"] += 1
                print(f"📦 CoinGecko кэш для {symbol}: ${price}")
                return price
        return None
    
    def _save_to_cache(self, symbol: str, price: float):
        self._cache[symbol] = (price, datetime.now())
    
    def get_stats(self) -> dict:
        cache_hit_rate = (
            self._stats["cache_hits"] / self._stats["total_requests"] * 100 
            if self._stats["total_requests"] > 0 else 0
        )
        return {
            **self._stats, 
            "cache_hit_rate": f"{cache_hit_rate:.1f}%", 
            "cache_size": len(self._cache)
        }
    
    async def get_price_usd(self, symbol: str, use_cache: bool = True) -> Optional[float]:
        symbol = symbol.upper().strip()
        print(f"🔍 Запрос CoinGecko для {symbol}...")
        self._stats["total_requests"] += 1
        
        if use_cache:
            cached_price = self._get_from_cache(symbol)
            if cached_price is not None:
                return cached_price
        
        if symbol not in self.COINGECKO_IDS:
            print(f"❌ CoinGecko: {symbol} не поддерживается")
            return None
        
        url = f"{self.BASE_URL}/simple/price"
        params = {"ids": self.COINGECKO_IDS[symbol], "vs_currencies": "usd"}
        
        try:
            await self._wait_for_rate_limit()
            session = await self._get_session()
            self._stats["api_calls"] += 1
            
            print(f"🔍 Запрос к CoinGecko: {url} с params={params}")
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', '60'))
                    print(f"⏳ CoinGecko rate limit, жду {retry_after} секунд")
                    await asyncio.sleep(retry_after)
                    return await self.get_price_usd(symbol, use_cache=False)
                
                print(f"🔍 CoinGecko статус: {response.status}")
                response.raise_for_status()
                data = await response.json()
                
                coin_id = self.COINGECKO_IDS[symbol]
                if coin_id not in data or "usd" not in data[coin_id]:
                    print(f"❌ CoinGecko: нет цены для {symbol} ({coin_id})")
                    print(f"🔍 Ответ: {data}")
                    return None
                
                price = data[coin_id]["usd"]
                print(f"✅ CoinGecko цена для {symbol}: ${price}")
                
                if use_cache:
                    self._save_to_cache(symbol, price)
                return price
        except Exception as e:
            print(f"❌ Ошибка получения цены {symbol}: {e}")
            self._stats["errors"].append(f"{symbol}: {str(e)}")
            return None
    
    @classmethod
    def is_supported(cls, symbol: str) -> bool:
        supported = symbol.upper().strip() in cls.COINGECKO_IDS
        print(f"🔍 CoinGecko поддерживает {symbol}: {supported}")
        return supported
    
    @classmethod
    def get_supported_symbols(cls) -> list:
        return sorted(cls.COINGECKO_IDS.keys())


# =============================================================================
# CONFIGURATION
# =============================================================================

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("❌ Не установлен токен бота! Создайте .env файл с BOT_TOKEN=ваш_токен")

CRYPTORANK_API_KEY = os.getenv("CRYPTORANK_API_KEY", "")

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher(storage=MemoryStorage(), fsm_strategy=FSMStrategy.USER_IN_CHAT)

# Initialize price fetchers
cryptorank_fetcher = CryptoRankPriceFetcher(api_key=CRYPTORANK_API_KEY)
coingecko_fetcher = CoinGeckoPriceFetcher(cache_ttl=300, max_requests_per_minute=5)


# =============================================================================
# FSM STATES - НОВЫЙ ПОРЯДОК
# =============================================================================

class Calc(StatesGroup):
    """Состояния для расчета позиции"""
    supply_ticker = State()         # Тикер залога
    borrow_ticker = State()         # Тикер займа
    supply_amount = State()         # Количество залога
    choose_price = State()          # Выбор источника цены
    supply_price_manual = State()   # Ручной ввод цены залога
    max_ltv = State()               # Maximum LTV (ПЕРВЫЙ параметр!)
    lt = State()                    # Liquidation Threshold (ВТОРОЙ параметр!)
    mode = State()                  # Режим расчета (ТРЕТИЙ!)
    ltv = State()                   # LTV (если режим по LTV)
    borrow = State()                # Сумма займа (если режим по сумме)


# =============================================================================
# KEYBOARDS
# =============================================================================

def price_choice_kb(cr_price: Optional[float], cg_price: Optional[float]):
    """Клавиатура выбора источника цены"""
    buttons = []
    
    # Сначала CoinGecko (более надежный)
    if cg_price is not None:
        if cg_price >= 1:
            price_str = f"${cg_price:,.2f}"
        elif cg_price >= 0.01:
            price_str = f"${cg_price:.4f}"
        else:
            price_str = f"${cg_price:.8f}"
        
        buttons.append([InlineKeyboardButton(
            text=f"🦎 CoinGecko: {price_str}",
            callback_data="price_coingecko"
        )])
    
    # Затем CryptoRank
    if cr_price is not None:
        if cr_price >= 1:
            price_str = f"${cr_price:,.2f}"
        elif cr_price >= 0.01:
            price_str = f"${cr_price:.4f}"
        else:
            price_str = f"${cr_price:.8f}"
        
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
    """
    Рассчитывает цену ликвидации
    При этой цене залога позиция будет ликвидирована
    """
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
    """Формирует итоговое сообщение с результатами"""
    status, emoji = get_position_status(calculations['hf'])
    price_source = data.get('supply_price_source', 'manual')
    
    # Умное форматирование цены (больше знаков для маленьких цен)
    price = calculations['price']
    if price >= 1:
        price_str = f"${price:,.2f}"
    elif price >= 0.01:
        price_str = f"${price:.4f}"
    else:
        price_str = f"${price:.8f}"
    
    # Аналогично для цены ликвидации
    liq_price = calculations['liq_price']
    if liq_price >= 1:
        liq_price_str = f"${liq_price:,.2f}"
    elif liq_price >= 0.01:
        liq_price_str = f"${liq_price:.4f}"
    else:
        liq_price_str = f"${liq_price:.8f}"
    
    # Определяем, как показывать цену
    source_names = {
        "cryptorank": "CryptoRank",
        "coingecko": "CoinGecko",
        "auto": "CoinGecko",  # backward compatibility
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
    
    # Цена ликвидации с указанием источника цены
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
        # Умное форматирование для цен сценариев
        if new_price >= 1:
            new_price_str = f"${new_price:,.2f}"
        elif new_price >= 0.01:
            new_price_str = f"${new_price:.4f}"
        else:
            new_price_str = f"${new_price:.8f}"
        result += f"• -{drop}% ({new_price_str}) → HF: {format_number(scen_hf, 2)}\n"
    
    # Рекомендации
    if calculations['hf'] < 1.3:
        result += (
            "\n<b>⚠️ РЕКОМЕНДАЦИИ:</b>\n"
            "• Увеличьте залог для повышения HF\n"
            "• Уменьшите сумму займа\n"
            "• Подготовьте средства для пополнения\n"
            "• Установите алерты на изменение цены"
        )
    
    # Уведомление о ручном вводе
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
    """Начало работы"""
    await state.clear()
    
    cr_status = "✅" if cryptorank_fetcher.is_available() else "❌"
    cg_supported = coingecko_fetcher.get_supported_symbols()
    
    await msg.answer(
        "🤖 <b>DeFi Position Calculator v2.3</b>\n"
        "<i>Калькулятор кредитных позиций в DeFi</i>\n\n"
        
        f"<b>📡 Источники цен:</b>\n"
        f"{cr_status} CryptoRank API\n"
        f"✅ CoinGecko API ({len(cg_supported)} монет)\n"
        f"✅ Ручной ввод (любые токены)\n\n"
        
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Введите <b>тикер залогового актива</b>\n"
        "(например: ETH, BTC, SOL)"
    )
    await state.set_state(Calc.supply_ticker)


@dp.message(Command("reset", "cancel"))
async def reset_cmd(msg: types.Message, state: FSMContext):
    """Сброс расчета"""
    await state.clear()
    await msg.answer("✅ Расчет сброшен. Используйте /start для нового расчета")


@dp.message(Command("help"))
async def help_cmd(msg: types.Message):
    """Справка"""
    await msg.answer(
        "<b>📖 Справка</b>\n\n"
        "<b>Команды:</b>\n"
        "/start - начать расчет\n"
        "/reset - сбросить расчет\n"
        "/supported - список монет\n"
        "/stats - статистика API\n"
        "/debug - отладочная информация\n\n"
        
        "<b>Порядок ввода:</b>\n"
        "1️⃣ Тикер залога\n"
        "2️⃣ Тикер займа\n"
        "3️⃣ Количество залога\n"
        "4️⃣ Цена (авто/ручная)\n"
        "5️⃣ Maximum LTV\n"
        "6️⃣ Liquidation Threshold\n"
        "7️⃣ Режим расчета\n"
        "8️⃣ LTV или сумма займа"
    )


@dp.message(Command("supported"))
async def supported_cmd(msg: types.Message):
    """Список поддерживаемых монет"""
    supported = coingecko_fetcher.get_supported_symbols()
    cols = 4
    rows = []
    for i in range(0, len(supported), cols):
        row = " | ".join(f"<code>{coin}</code>" for coin in supported[i:i+cols])
        rows.append(row)
    
    cr_status = "настроен ✅" if cryptorank_fetcher.is_available() else "не настроен ❌"
    
    await msg.answer(
        f"<b>📡 Источники цен:</b>\n\n"
        f"<b>CryptoRank API:</b> {cr_status}\n"
        f"(поддерживает большинство токенов)\n\n"
        f"<b>CoinGecko API ({len(supported)} монет):</b>\n"
        + "\n".join(rows) + 
        "\n\n💡 <i>Для остальных - ручной ввод</i>"
    )


@dp.message(Command("stats"))
async def stats_cmd(msg: types.Message):
    """Статистика API"""
    cg_stats = coingecko_fetcher.get_stats()
    cr_stats = cryptorank_fetcher.get_stats()
    
    stats_text = (
        f"<b>📊 Статистика API</b>\n\n"
        f"<b>CoinGecko:</b>\n"
        f"Запросов: {cg_stats['total_requests']}\n"
        f"API вызовов: {cg_stats['api_calls']}\n"
        f"Из кэша: {cg_stats['cache_hits']}\n"
        f"Процент кэша: {cg_stats['cache_hit_rate']}\n\n"
        f"<b>CryptoRank:</b>\n"
        f"Запросов: {cr_stats['total']}\n"
        f"Успешных: {cr_stats['success']}\n"
        f"Ошибок: {cr_stats['fail']}\n"
        f"Успешность: {cr_stats.get('success_rate', '0%')}\n"
    )
    
    if cr_stats.get('recent_errors'):
        stats_text += f"\n<b>Последние ошибки CryptoRank:</b>\n"
        for error in cr_stats['recent_errors']:
            stats_text += f"• {error[:50]}...\n"
    
    await msg.answer(stats_text)


@dp.message(Command("debug"))
async def debug_cmd(msg: types.Message):
    """Отладочная информация"""
    cr_available = cryptorank_fetcher.is_available()
    cr_key_preview = "***" + CRYPTORANK_API_KEY[-4:] if CRYPTORANK_API_KEY and len(CRYPTORANK_API_KEY) > 4 else "не установлен"
    
    await msg.answer(
        f"<b>🐛 Отладочная информация</b>\n\n"
        f"<b>CryptoRank:</b>\n"
        f"Доступен: {'✅' if cr_available else '❌'}\n"
        f"Ключ: {cr_key_preview}\n"
        f"Длина ключа: {len(CRYPTORANK_API_KEY) if CRYPTORANK_API_KEY else 0}\n\n"
        f"<b>CoinGecko:</b>\n"
        f"Доступен: ✅\n"
        f"Поддерживаемых монет: {len(coingecko_fetcher.get_supported_symbols())}\n\n"
        f"<i>Для теста попробуйте тикер BTC</i>"
    )


# =============================================================================
# STATE HANDLERS - НОВЫЙ ПОРЯДОК ВВОДА
# =============================================================================

@dp.message(Calc.supply_ticker)
async def process_supply_ticker(msg: types.Message, state: FSMContext):
    """Тикер залога"""
    valid, ticker, error = validate_ticker(msg.text)
    if not valid:
        await msg.answer(f"❌ {error}\n\nВведите корректный тикер:")
        return
    
    await state.update_data(supply_ticker=ticker)
    is_supported = coingecko_fetcher.is_supported(ticker)
    
    await msg.answer(
        f"✅ <b>Залоговый актив:</b> {ticker}\n"
        f"{'🌐' if is_supported else '✍️'} Цена: {'автоматическая' if is_supported else 'ручной ввод'}\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Введите <b>тикер заимствуемого актива</b>"
    )
    await state.set_state(Calc.borrow_ticker)


@dp.message(Calc.borrow_ticker)
async def process_borrow_ticker(msg: types.Message, state: FSMContext):
    """Тикер займа"""
    valid, ticker, error = validate_ticker(msg.text)
    if not valid:
        await msg.answer(f"❌ {error}\n\nВведите корректный тикер:")
        return
    
    await state.update_data(borrow_ticker=ticker)
    data = await state.get_data()
    
    await msg.answer(
        f"✅ <b>Заимствуемый актив:</b> {ticker}\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"Введите <b>количество {data['supply_ticker']}</b>"
    )
    await state.set_state(Calc.supply_amount)


@dp.message(Calc.supply_amount)
async def process_supply_amount(msg: types.Message, state: FSMContext):
    """Количество залога"""
    valid, value, error = validate_number(msg.text, min_val=0.000001)
    if not valid:
        await msg.answer(f"❌ {error}\n\nВведите количество:")
        return
    
    await state.update_data(supply_amount=value)
    data = await state.get_data()
    ticker = data['supply_ticker']
    
    await msg.answer(f"✅ Количество: {value:.6f}\n\n⏳ Получаю цены {ticker}...")
    
    # Пытаемся получить цены из обоих источников ПАРАЛЛЕЛЬНО
    print(f"\n{'='*60}")
    print(f"🔍 ПОЛУЧЕНИЕ ЦЕН ДЛЯ {ticker}")
    print(f"{'='*60}")
    
    # Запускаем оба запроса параллельно
    cr_task = asyncio.create_task(cryptorank_fetcher.get_price_usd(ticker))
    cg_task = asyncio.create_task(coingecko_fetcher.get_price_usd(ticker))
    
    cr_price, cg_price = await asyncio.gather(cr_task, cg_task)
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ДЛЯ {ticker}:")
    print(f"CryptoRank: ${cr_price if cr_price else 'нет'}")
    print(f"CoinGecko: ${cg_price if cg_price else 'нет'}")
    print(f"{'='*60}\n")
    
    # Если есть хотя бы одна цена - предлагаем выбор
    if cr_price is not None or cg_price is not None:
        await state.update_data(cryptorank_price=cr_price, coingecko_price=cg_price)
        
        sources = []
        if cg_price:
            price_str = f"${cg_price:,.2f}" if cg_price >= 1 else f"${cg_price:.6f}"
            sources.append(f"🦎 CoinGecko: {price_str}")
        if cr_price:
            price_str = f"${cr_price:,.2f}" if cr_price >= 1 else f"${cr_price:.6f}"
            sources.append(f"✅ CryptoRank: {price_str}")
        
        if sources:
            await msg.answer(
                f"💱 <b>Найдены цены {ticker}:</b>\n" +
                "\n".join(f"• {s}" for s in sources) +
                "\n\n<b>Выберите источник:</b>",
                reply_markup=price_choice_kb(cr_price, cg_price)
            )
            await state.set_state(Calc.choose_price)
        else:
            # Нет автоматических цен - запрашиваем ручной ввод
            await msg.answer(
                f"❌ Цена {ticker} не найдена в API\n\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Введите <b>цену {ticker}</b> в USD вручную:"
            )
            await state.set_state(Calc.supply_price_manual)
    else:
        # Нет автоматических цен - запрашиваем ручной ввод
        await msg.answer(
            f"❌ Цена {ticker} не найдена в API\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Введите <b>цену {ticker}</b> в USD вручную:"
        )
        await state.set_state(Calc.supply_price_manual)


@dp.callback_query(F.data.startswith("price_"))
async def process_price_choice(cb: types.CallbackQuery, state: FSMContext):
    """Обработка выбора источника цены"""
    await cb.answer()
    
    data = await state.get_data()
    choice = cb.data.split("_")[1]  # cryptorank, coingecko, manual
    
    if choice == "manual":
        await cb.message.edit_text(
            f"✏️ Введите цену <b>{data['supply_ticker']}</b> в USD вручную:"
        )
        await state.set_state(Calc.supply_price_manual)
        return
    
    # Используем выбранную API цену
    if choice == "cryptorank":
        price = data.get('cryptorank_price')
        source = "cryptorank"
        source_name = "CryptoRank"
    else:  # coingecko
        price = data.get('coingecko_price')
        source = "coingecko"
        source_name = "CoinGecko"
    
    if price is None:
        await cb.message.edit_text("❌ Ошибка получения цены. Введите вручную:")
        await state.set_state(Calc.supply_price_manual)
        return
    
    await state.update_data(supply_price=price, supply_price_source=source)
    
    supply_amount = data['supply_amount']
    collateral_value = supply_amount * price
    
    # Умное форматирование
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
    """Ручной ввод цены"""
    valid, price, error = validate_number(msg.text, min_val=0.000001)
    if not valid:
        await msg.answer(f"❌ {error}\n\nВведите цену:")
        return
    
    data = await state.get_data()
    ticker = data['supply_ticker']
    amount = data['supply_amount']
    
    await state.update_data(supply_price=price, supply_price_source="manual")
    collateral_value = amount * price
    
    # Умное форматирование цены
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
    """Maximum LTV - ПЕРВЫЙ параметр"""
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    if not valid:
        await msg.answer(f"❌ {error}\n\nMax LTV должен быть 0-100%. Введите:")
        return
    
    await state.update_data(max_ltv=value / 100)
    
    # Получаем данные для расчёта максимального займа
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
    """Liquidation Threshold - ВТОРОЙ параметр"""
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    if not valid:
        await msg.answer(f"❌ {error}\n\nLT должен быть 0-100%. Введите:")
        return
    
    data = await state.get_data()
    max_ltv = data.get('max_ltv', 0) * 100
    
    # Проверка: LT должен быть >= Max LTV
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
    """Режим расчета - ТРЕТИЙ выбор"""
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
        # Рассчитываем максимально возможную сумму займа
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
    """LTV для расчета"""
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    if not valid:
        await msg.answer(f"❌ {error}\n\nLTV должен быть 0-100%. Введите:")
        return
    
    data = await state.get_data()
    max_ltv = data.get('max_ltv', 0) * 100
    
    # Проверка: LTV должен быть <= Max LTV
    if value > max_ltv:
        await msg.answer(
            f"❌ <b>Ошибка:</b> LTV ({value}%) не может превышать "
            f"Maximum LTV ({max_ltv:.0f}%)\n\n"
            "Введите корректное значение:"
        )
        return
    
    await state.update_data(ltv=value / 100)
    
    # Переходим к расчету
    await calculate_position(msg, state)


@dp.message(Calc.borrow)
async def process_borrow(msg: types.Message, state: FSMContext):
    """Сумма займа"""
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
    
    # Проверка: займ не должен превышать максимально возможный
    if value > max_borrow_allowed:
        await msg.answer(
            f"❌ <b>Ошибка:</b> Сумма займа ({format_currency(value)}) превышает "
            f"максимально возможный займ ({format_currency(max_borrow_allowed)}) "
            f"при Max LTV {max_ltv*100:.0f}%\n\n"
            "Введите корректную сумму:"
        )
        return
    
    await state.update_data(borrow=value)
    
    # Переходим к расчету
    await calculate_position(msg, state)


# =============================================================================
# CALCULATION
# =============================================================================

async def calculate_position(msg: types.Message, state: FSMContext):
    """Финальный расчет"""
    try:
        data = await state.get_data()
        
        # Проверка данных
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
        
        # Расчет займа и LTV
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
        
        # Собираем результаты
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
        
        # Отправка результата
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
    """Обработчик неизвестных сообщений"""
    current_state = await state.get_state()
    if current_state:
        await msg.answer("⚠️ Следуйте инструкциям или используйте /reset")
    else:
        await msg.answer("👋 Привет! Используйте /start для начала расчета")


@dp.error()
async def error_handler(event, exception):
    """Глобальный обработчик ошибок"""
    print(f"❌ Глобальная ошибка: {exception}")
    import traceback
    traceback.print_exc()
    return True


# =============================================================================
# STARTUP & SHUTDOWN
# =============================================================================

async def on_startup():
    print("\n" + "=" * 70)
    print("🚀 DeFi Position Calculator Bot v2.3")
    print("=" * 70)
    
    bot_info = await bot.get_me()
    print(f"✅ Бот: @{bot_info.username}")
    
    # Проверка CryptoRank
    if cryptorank_fetcher.is_available():
        key_preview = CRYPTORANK_API_KEY[:4] + "..." + CRYPTORANK_API_KEY[-4:] if len(CRYPTORANK_API_KEY) > 8 else "***"
        print(f"✅ CryptoRank API: настроен (ключ: {key_preview})")
        
        # Тестовый запрос для проверки
        print(f"🔍 Тестовый запрос CryptoRank для BTC...")
        test_price = await cryptorank_fetcher.get_price_usd("BTC")
        if test_price:
            print(f"✅ CryptoRank работает (BTC: ${test_price:,.2f})")
        else:
            print(f"❌ CryptoRank тестовый запрос не удался")
    else:
        print("ℹ️  CryptoRank API: не настроен (опционально)")
    
    # Проверка CoinGecko
    test_price = await coingecko_fetcher.get_price_usd("BTC")
    if test_price:
        print(f"✅ CoinGecko работает (BTC: ${test_price:,.2f})")
        print(f"✅ CoinGecko: {len(coingecko_fetcher.get_supported_symbols())} монет")
    else:
        print(f"❌ CoinGecko тестовый запрос не удался")
    
    print("✅ Новый порядок: Max LTV → LT → режим расчета")
    print("=" * 70)
    print("✅ БОТ ГОТОВ")
    print("=" * 70 + "\n")


async def on_shutdown():
    await cryptorank_fetcher.close()
    await coingecko_fetcher.close()
    await bot.session.close()
    print("\n👋 Бот остановлен")


async def main():
    try:
        await on_startup()
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except KeyboardInterrupt:
        print("\n⚠️ Остановка...")
    finally:
        await on_shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 До свидания!")
