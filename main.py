import aiohttp
import asyncio
from typing import Optional, Dict
from datetime import datetime, timedelta
from collections import deque

class CoinGeckoPriceFetcher:
    """
    Класс для получения цен криптовалют через CoinGecko API
    с кэшированием, rate limiting и retry механизмом
    """
    
    # Маппинг тикеров на CoinGecko ID
    COINGECKO_IDS = {
        "ETH": "ethereum",
        "BTC": "bitcoin",
        "SOL": "solana",
        "USDC": "usd-coin",
        "USDT": "tether",
        "USDD": "usdd",
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
        "SHIB": "shiba-inu"
    }
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(
        self, 
        cache_ttl: int = 60,
        max_requests_per_minute: int = 10,  # CoinGecko free: ~10-15/min
        enable_retry: bool = True,
        max_retries: int = 3
    ):
        """
        Инициализация фетчера
        
        Args:
            cache_ttl: Время жизни кэша в секундах (по умолчанию 60)
            max_requests_per_minute: Максимум запросов в минуту
            enable_retry: Включить retry при ошибках
            max_retries: Максимум попыток повтора
        """
        self._cache: Dict[str, tuple[float, datetime]] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl)
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self._max_requests_per_minute = max_requests_per_minute
        self._request_times = deque(maxlen=max_requests_per_minute)
        self._rate_limit_lock = asyncio.Lock()
        
        # Retry настройки
        self._enable_retry = enable_retry
        self._max_retries = max_retries
        
        # Статистика
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "rate_limit_waits": 0,
            "errors": 0
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Получает или создает aiohttp сессию"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=15)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Закрывает HTTP сессию"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _wait_for_rate_limit(self):
        """Ожидает, если достигнут лимит запросов"""
        async with self._rate_limit_lock:
            now = datetime.now()
            
            # Удаляем старые запросы (старше 1 минуты)
            while self._request_times and (now - self._request_times[0]).total_seconds() > 60:
                self._request_times.popleft()
            
            # Если достигнут лимит, ждем
            if len(self._request_times) >= self._max_requests_per_minute:
                oldest_request = self._request_times[0]
                wait_time = 60 - (now - oldest_request).total_seconds()
                
                if wait_time > 0:
                    self._stats["rate_limit_waits"] += 1
                    print(f"⏳ Rate limit: ожидание {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time + 0.5)  # +0.5s буфер
            
            # Регистрируем новый запрос
            self._request_times.append(now)
    
    def _get_from_cache(self, symbol: str) -> Optional[float]:
        """Получает цену из кэша, если она не устарела"""
        if symbol in self._cache:
            price, timestamp = self._cache[symbol]
            if datetime.now() - timestamp < self._cache_ttl:
                self._stats["cache_hits"] += 1
                return price
        return None
    
    def _save_to_cache(self, symbol: str, price: float):
        """Сохраняет цену в кэш"""
        self._cache[symbol] = (price, datetime.now())
    
    def clear_cache(self):
        """Очищает весь кэш"""
        self._cache.clear()
    
    def get_stats(self) -> dict:
        """Возвращает статистику использования"""
        cache_hit_rate = (
            self._stats["cache_hits"] / self._stats["total_requests"] * 100 
            if self._stats["total_requests"] > 0 else 0
        )
        return {
            **self._stats,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "cache_size": len(self._cache)
        }
    
    async def _make_request(
        self, 
        url: str, 
        params: dict, 
        retry_count: int = 0
    ) -> Optional[dict]:
        """
        Выполняет HTTP запрос с retry логикой
        
        Args:
            url: URL для запроса
            params: Параметры запроса
            retry_count: Текущая попытка
            
        Returns:
            JSON ответ или None
        """
        try:
            # Ждем если нужно (rate limiting)
            await self._wait_for_rate_limit()
            
            session = await self._get_session()
            self._stats["api_calls"] += 1
            
            async with session.get(url, params=params) as response:
                # Обрабатываем 429 специально
                if response.status == 429:
                    retry_after = response.headers.get('Retry-After', '60')
                    wait_time = int(retry_after) if retry_after.isdigit() else 60
                    
                    if self._enable_retry and retry_count < self._max_retries:
                        print(f"⚠️ 429 Too Many Requests. Ожидание {wait_time}s перед повтором...")
                        await asyncio.sleep(wait_time)
                        return await self._make_request(url, params, retry_count + 1)
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=429,
                            message="Too Many Requests - лимит API исчерпан"
                        )
                
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                raise  # 429 уже обработали выше
            
            # Для других HTTP ошибок
            if self._enable_retry and retry_count < self._max_retries and e.status >= 500:
                wait_time = 2 ** retry_count  # Экспоненциальная задержка
                print(f"⚠️ HTTP {e.status}: повтор через {wait_time}s...")
                await asyncio.sleep(wait_time)
                return await self._make_request(url, params, retry_count + 1)
            
            self._stats["errors"] += 1
            raise
            
        except aiohttp.ClientError as e:
            # Сетевые ошибки
            if self._enable_retry and retry_count < self._max_retries:
                wait_time = 2 ** retry_count
                print(f"⚠️ Сетевая ошибка: повтор через {wait_time}s...")
                await asyncio.sleep(wait_time)
                return await self._make_request(url, params, retry_count + 1)
            
            self._stats["errors"] += 1
            raise
    
    async def get_price_usd(
        self, 
        symbol: str, 
        use_cache: bool = True
    ) -> Optional[float]:
        """
        Получает цену криптовалюты в USD
        
        Args:
            symbol: Тикер криптовалюты (например: ETH, BTC)
            use_cache: Использовать ли кэш
            
        Returns:
            Цена в USD или None в случае ошибки
        """
        symbol = symbol.upper().strip()
        self._stats["total_requests"] += 1
        
        # Проверяем кэш
        if use_cache:
            cached_price = self._get_from_cache(symbol)
            if cached_price is not None:
                return cached_price
        
        # Проверяем поддержку тикера
        if symbol not in self.COINGECKO_IDS:
            print(f"⚠️ Тикер '{symbol}' не поддерживается")
            return None
        
        # Формируем запрос
        url = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": self.COINGECKO_IDS[symbol],
            "vs_currencies": "usd"
        }
        
        try:
            data = await self._make_request(url, params)
            
            if data is None:
                return None
            
            coin_id = self.COINGECKO_IDS[symbol]
            if coin_id not in data or "usd" not in data[coin_id]:
                print(f"❌ Неожиданный формат ответа для {symbol}")
                return None
            
            price = data[coin_id]["usd"]
            
            # Сохраняем в кэш
            if use_cache:
                self._save_to_cache(symbol, price)
            
            return price
            
        except aiohttp.ClientResponseError as e:
            print(f"❌ HTTP ошибка при получении цены {symbol}: {e.status} - {e.message}")
            return None
        except aiohttp.ClientError as e:
            print(f"❌ Сетевая ошибка при получении цены {symbol}: {e}")
            return None
        except Exception as e:
            print(f"❌ Неожиданная ошибка при получении цены {symbol}: {e}")
            self._stats["errors"] += 1
            return None
    
    async def get_batch_prices(
        self, 
        symbols: list[str], 
        use_cache: bool = True
    ) -> Dict[str, Optional[float]]:
        """
        Получает цены для нескольких криптовалют одним запросом
        
        Args:
            symbols: Список тикеров
            use_cache: Использовать ли кэш
            
        Returns:
            Словарь {тикер: цена}
        """
        symbols = [s.upper().strip() for s in symbols]
        result = {}
        symbols_to_fetch = []
        
        # Проверяем кэш
        for symbol in symbols:
            self._stats["total_requests"] += 1
            
            if use_cache:
                cached_price = self._get_from_cache(symbol)
                if cached_price is not None:
                    result[symbol] = cached_price
                    continue
            
            if symbol not in self.COINGECKO_IDS:
                print(f"⚠️ Тикер '{symbol}' не поддерживается")
                result[symbol] = None
                continue
            
            symbols_to_fetch.append(symbol)
        
        # Если все в кэше, возвращаем результат
        if not symbols_to_fetch:
            return result
        
        # Формируем batch запрос
        coin_ids = [self.COINGECKO_IDS[s] for s in symbols_to_fetch]
        url = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": "usd"
        }
        
        try:
            data = await self._make_request(url, params)
            
            if data is None:
                for symbol in symbols_to_fetch:
                    result[symbol] = None
                return result
            
            for symbol in symbols_to_fetch:
                coin_id = self.COINGECKO_IDS[symbol]
                if coin_id in data and "usd" in data[coin_id]:
                    price = data[coin_id]["usd"]
                    result[symbol] = price
                    if use_cache:
                        self._save_to_cache(symbol, price)
                else:
                    result[symbol] = None
                    
        except Exception as e:
            print(f"❌ Ошибка при batch запросе: {e}")
            for symbol in symbols_to_fetch:
                result[symbol] = None
        
        return result
    
    @classmethod
    def is_supported(cls, symbol: str) -> bool:
        """Проверяет, поддерживается ли тикер"""
        return symbol.upper().strip() in cls.COINGECKO_IDS
    
    @classmethod
    def get_supported_symbols(cls) -> list[str]:
        """Возвращает список поддерживаемых тикеров"""
        return list(cls.COINGECKO_IDS.keys())


# ---------- ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ----------
async def example_with_rate_limiting():
    """Пример с rate limiting"""
    print("=== Пример с rate limiting ===\n")
    
    # Создаем fetcher с ограничением 5 запросов в минуту
    fetcher = CoinGeckoPriceFetcher(
        cache_ttl=120,  # Кэш на 2 минуты
        max_requests_per_minute=5,  # Только 5 запросов/минуту
        enable_retry=True,
        max_retries=2
    )
    
    try:
        # Получаем несколько цен - некоторые будут ждать
        symbols = ["BTC", "ETH", "SOL", "BNB", "ADA", "DOT"]
        
        print(f"Запрашиваю цены для {len(symbols)} монет...")
        print(f"Лимит: {fetcher._max_requests_per_minute} запросов/минуту\n")
        
        # Используем batch запрос - это 1 API call вместо 6!
        prices = await fetcher.get_batch_prices(symbols)
        
        print("\n📊 Результаты:")
        for symbol, price in prices.items():
            if price:
                print(f"  ✅ {symbol}: ${price:,.2f}")
            else:
                print(f"  ❌ {symbol}: недоступно")
        
        # Показываем статистику
        stats = fetcher.get_stats()
        print(f"\n📈 Статистика:")
        print(f"  Всего запросов: {stats['total_requests']}")
        print(f"  API вызовов: {stats['api_calls']}")
        print(f"  Попаданий в кэш: {stats['cache_hits']}")
        print(f"  Процент кэша: {stats['cache_hit_rate']}")
        print(f"  Ожиданий rate limit: {stats['rate_limit_waits']}")
        print(f"  Ошибок: {stats['errors']}")
        
        # Повторный запрос - всё из кэша!
        print("\n🔄 Повторный запрос тех же монет...")
        prices2 = await fetcher.get_batch_prices(symbols)
        
        stats2 = fetcher.get_stats()
        print(f"  Процент кэша: {stats2['cache_hit_rate']} (было {stats['cache_hit_rate']})")
        
    finally:
        await fetcher.close()


async def example_conservative():
    """Консервативный пример для бота (минимум запросов)"""
    print("\n=== Консервативный режим для бота ===\n")
    
    # Очень консервативные настройки
    fetcher = CoinGeckoPriceFetcher(
        cache_ttl=300,  # Кэш на 5 минут
        max_requests_per_minute=3,  # Только 3 запроса/минуту
        enable_retry=True,
        max_retries=3
    )
    
    try:
        # Получаем популярные монеты
        symbols = ["BTC", "ETH", "USDC"]
        
        print("Получаю цены популярных монет...")
        prices = await fetcher.get_batch_prices(symbols)
        
        for symbol, price in prices.items():
            if price:
                print(f"✅ {symbol}: ${price:,.2f}")
        
        # Ждем немного
        print("\n⏳ Ожидание 5 секунд...")
        await asyncio.sleep(5)
        
        # Запрашиваем снова - будет из кэша
        print("Повторный запрос...")
        prices2 = await fetcher.get_batch_prices(symbols)
        
        stats = fetcher.get_stats()
        print(f"\n📊 API вызовов: {stats['api_calls']} (должен быть 1)")
        print(f"📊 Попаданий в кэш: {stats['cache_hits']}")
        
    finally:
        await fetcher.close()


async def main():
    """Основная функция с примерами"""
    print("=" * 60)
    print("CoinGecko Price Fetcher с Rate Limiting")
    print("=" * 60 + "\n")
    
    await example_with_rate_limiting()
    await example_conservative()
    
    print("\n" + "=" * 60)
    print("✅ Все примеры выполнены")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
