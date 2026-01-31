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

# ---------- CONFIGURATION ----------
TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    raise ValueError(
        "Не установлен токен бота. Установите переменную окружения BOT_TOKEN"
    )

bot = Bot(
    token=TOKEN,
    default=DefaultBotProperties(parse_mode="HTML")
)
dp = Dispatcher(storage=MemoryStorage(), fsm_strategy=FSMStrategy.USER_IN_CHAT)

# ---------- DATA CLASSES ----------
@dataclass
class PositionData:
    """Класс для хранения данных позиции"""
    supply_ticker: str
    borrow_ticker: str
    supply_amount: float
    supply_price: float
    lt: float
    max_ltv: float
    ltv: Optional[float] = None
    borrow: Optional[float] = None
    
    @property
    def collateral_value(self) -> float:
        return self.supply_amount * self.supply_price
    
    def get_ltv(self) -> float:
        """Возвращает LTV в зависимости от режима"""
        if self.ltv is not None:
            return self.ltv
        return self.borrow / self.collateral_value if self.collateral_value > 0 else 0
    
    def get_borrow_amount(self) -> float:
        """Возвращает сумму займа в зависимости от режима"""
        if self.borrow is not None:
            return self.borrow
        return self.collateral_value * self.ltv if self.ltv is not None else 0

# ---------- STATES ----------
class Calc(StatesGroup):
    supply_ticker = State()
    borrow_ticker = State()
    supply_amount = State()
    supply_price = State()
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

# ---------- VALIDATION HELPERS ----------
def validate_number(
    text: str, 
    min_val: float = 0, 
    max_val: Optional[float] = None
) -> Tuple[bool, float, str]:
    """
    Проверяет корректность числового ввода
    
    Args:
        text: Текст для валидации
        min_val: Минимальное допустимое значение
        max_val: Максимальное допустимое значение (опционально)
    
    Returns:
        Tuple[bool, float, str]: (валидно, значение, сообщение об ошибке)
    """
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
    """
    Проверяет корректность тикера
    
    Returns:
        Tuple[bool, str, str]: (валидно, тикер, сообщение об ошибке)
    """
    ticker = text.upper().strip()
    if len(ticker) > max_length:
        return False, "", f"Тикер слишком длинный (max {max_length} символов)"
    if not ticker.isalnum():
        return False, "", "Тикер должен содержать только буквы и цифры"
    return True, ticker, ""

def format_currency(value: float) -> str:
    """Форматирует денежные значения"""
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.2f}"

def format_number(value: float, decimals: int = 2) -> str:
    """Форматирует числа с заданным количеством десятичных знаков"""
    if value == float('inf'):
        return "∞"
    return f"{value:.{decimals}f}"

# ---------- CALCULATION HELPERS ----------
def calculate_health_factor(collateral: float, lt: float, borrow: float) -> float:
    """Рассчитывает Health Factor"""
    if borrow <= 0:
        return float('inf')
    return (collateral * lt) / borrow

def calculate_liquidation_price(borrow: float, supply_amount: float, lt: float) -> float:
    """Рассчитывает цену ликвидации"""
    denominator = supply_amount * lt
    if denominator <= 0:
        return 0
    return borrow / denominator

def get_position_status(hf: float) -> Tuple[str, str]:
    """
    Определяет статус позиции по Health Factor
    
    Returns:
        Tuple[str, str]: (статус с эмодзи, эмодзи)
    """
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
    """Обработчик команды /start"""
    await state.clear()
    await msg.answer(
        "<b>📊 DeFi Risk calculator </b>\n\n"
        "Введите тикер залогового актива (например: ETH, SOL, BTC):"
    )
    await state.set_state(Calc.supply_ticker)

@dp.message(Command("reset", "отмена", "сброс"))
async def reset_cmd(msg: types.Message, state: FSMContext):
    """Обработчик команды сброса"""
    await state.clear()
    await msg.answer(
        "✅ Состояние сброшено.\n"
        "Используйте /start для начала нового расчета."
    )

@dp.message(Command("help", "помощь"))
async def help_cmd(msg: types.Message):
    """Обработчик команды помощи"""
    await msg.answer(
        "<b>📖 Помощь по боту</b>\n\n"
        "<b>Команды:</b>\n"
        "• /start - начать расчет позиции\n"
        "• /reset - сбросить текущий расчет\n"
        "• /help - показать это сообщение\n\n"
        "<b>Что рассчитывает бот:</b>\n"
        "• Health Factor (фактор здоровья позиции)\n"
        "• Цену ликвидации\n"
        "• Максимальный возможный займ\n"
        "• Буфер безопасности\n"
        "• Сценарии при падении цены на 10%, 20%, 30%\n\n"
        "<b>Термины:</b>\n"
        "• LTV (Loan-to-Value) - отношение займа к залогу\n"
        "• LT (Liquidation Threshold) - порог ликвидации\n"
        "• HF (Health Factor) - когда HF < 1, позиция ликвидируется"
    )

# ---------- STATE HANDLERS ----------
@dp.message(Calc.supply_ticker)
async def process_supply_ticker(msg: types.Message, state: FSMContext):
    """Обработка тикера залогового актива"""
    valid, ticker, error = validate_ticker(msg.text)
    
    if not valid:
        await msg.answer(f"❌ {error}\n\nПожалуйста, введите корректный тикер:")
        return
    
    await state.update_data(supply_ticker=ticker)
    await msg.answer(
        f"✅ Залоговый актив: <b>{ticker}</b>\n\n"
        "Введите тикер заимствуемого актива (например: USDC, DAI, USDT):"
    )
    await state.set_state(Calc.borrow_ticker)

@dp.message(Calc.borrow_ticker)
async def process_borrow_ticker(msg: types.Message, state: FSMContext):
    """Обработка тикера заимствуемого актива"""
    valid, ticker, error = validate_ticker(msg.text)
    
    if not valid:
        await msg.answer(f"❌ {error}\n\nПожалуйста, введите корректный тикер:")
        return
    
    await state.update_data(borrow_ticker=ticker)
    await msg.answer(
        f"✅ Заимствуемый актив: <b>{ticker}</b>\n\n"
        "Введите количество залогового актива:"
    )
    await state.set_state(Calc.supply_amount)

@dp.message(Calc.supply_amount)
async def process_supply_amount(msg: types.Message, state: FSMContext):
    """Обработка количества залогового актива"""
    valid, value, error = validate_number(msg.text, min_val=0.000001)
    
    if not valid:
        await msg.answer(f"❌ {error}\n\nПожалуйста, введите количество:")
        return
    
    await state.update_data(supply_amount=value)
    data = await state.get_data()
    
    await msg.answer(
        f"✅ Залоговый актив: <b>{data.get('supply_ticker')}</b>\n"
        f"✅ Количество: <b>{value:.6f}</b>\n\n"
        "Введите цену залогового актива в USD:"
    )
    await state.set_state(Calc.supply_price)

@dp.message(Calc.supply_price)
async def process_supply_price(msg: types.Message, state: FSMContext):
    """Обработка цены залогового актива"""
    valid, value, error = validate_number(msg.text, min_val=0.000001)
    
    if not valid:
        await msg.answer(f"❌ {error}\n\nПожалуйста, введите цену:")
        return
    
    await state.update_data(supply_price=value)
    data = await state.get_data()
    
    supply_amount = data.get('supply_amount', 0)
    collateral_value = supply_amount * value
    
    await msg.answer(
        f"<b>📊 Предварительный расчет</b>\n\n"
        f"Залоговый актив: <b>{data.get('supply_ticker')}</b>\n"
        f"Количество: {supply_amount:.6f}\n"
        f"Цена: ${value:.2f}\n"
        f"<b>💰 Стоимость залога: {format_currency(collateral_value)}</b>\n\n"
        "Выберите режим расчета:",
        reply_markup=mode_kb
    )
    await state.set_state(Calc.mode)

@dp.callback_query(F.data.startswith("mode_"))
async def process_mode(cb: types.CallbackQuery, state: FSMContext):
    """Обработка выбора режима расчета"""
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
    else:  # mode_borrow
        await cb.message.edit_text(
            f"<b>💵 Режим: Расчет по сумме займа</b>\n\n"
            f"Стоимость залога: {format_currency(collateral_value)}\n\n"
            "Введите сумму займа в USD:"
        )
        await state.set_state(Calc.borrow)

@dp.message(Calc.ltv)
async def process_ltv(msg: types.Message, state: FSMContext):
    """Обработка LTV"""
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    
    if not valid:
        await msg.answer(
            f"❌ {error}\n\n"
            "LTV должен быть от 0 до 100%.\n"
            "Введите LTV (%):"
        )
        return
    
    await state.update_data(ltv=value / 100)
    
    data = await state.get_data()
    supply_amount = data.get('supply_amount', 0)
    supply_price = data.get('supply_price', 0)
    collateral_value = supply_amount * supply_price
    borrow_amount = collateral_value * (value / 100)
    
    await msg.answer(
        f"✅ <b>LTV: {value}%</b>\n"
        f"Сумма займа при таком LTV: {format_currency(borrow_amount)}\n\n"
        "Введите Liquidation Threshold (LT) в % (например: 75):"
    )
    await state.set_state(Calc.lt)

@dp.message(Calc.borrow)
async def process_borrow(msg: types.Message, state: FSMContext):
    """Обработка суммы займа"""
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
            "Введите корректную сумму займа:"
        )
        return
    
    await state.update_data(borrow=value)
    
    ltv_percent = (value / collateral_value) * 100 if collateral_value > 0 else 0
    
    await msg.answer(
        f"✅ <b>Сумма займа: {format_currency(value)}</b>\n"
        f"LTV при такой сумме: {ltv_percent:.1f}%\n\n"
        "Введите Liquidation Threshold (LT) в % (например: 75):"
    )
    await state.set_state(Calc.lt)

@dp.message(Calc.lt)
async def process_lt(msg: types.Message, state: FSMContext):
    """Обработка Liquidation Threshold"""
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    
    if not valid:
        await msg.answer(
            f"❌ {error}\n\n"
            "LT должен быть от 0 до 100%.\n"
            "Введите LT (%):"
        )
        return
    
    await state.update_data(lt=value / 100)
    
    await msg.answer(
        f"✅ <b>Liquidation Threshold: {value}%</b>\n\n"
        "Введите Maximum LTV в % (например: 65):"
    )
    await state.set_state(Calc.max_ltv)

# ---------- CALCULATION ----------
@dp.message(Calc.max_ltv)
async def calculate_position(msg: types.Message, state: FSMContext):
    """Основной расчет позиции"""
    try:
        # Валидация Max LTV
        valid, max_ltv_input, error = validate_number(msg.text, min_val=0, max_val=100)
        if not valid:
            await msg.answer(f"❌ {error}\n\nВведите Maximum LTV (%):")
            return
        
        max_ltv = max_ltv_input / 100
        
        # Получаем все данные
        data = await state.get_data()
        
        # Проверяем обязательные поля
        required_fields = ['supply_ticker', 'borrow_ticker', 'supply_amount', 
                          'supply_price', 'lt', 'mode']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            await msg.answer(
                f"❌ Отсутствуют данные: {', '.join(missing_fields)}\n\n"
                "Пожалуйста, начните заново с /start"
            )
            await state.clear()
            return
        
        # Извлекаем данные
        supply_amt = data['supply_amount']
        price = data['supply_price']
        lt = data['lt']
        mode = data['mode']
        
        # Рассчитываем стоимость залога
        collateral = supply_amt * price
        
        # Рассчитываем займ и LTV в зависимости от режима
        if mode == "mode_ltv":
            if 'ltv' not in data:
                await msg.answer("❌ Отсутствует LTV\n\nНачните заново с /start")
                await state.clear()
                return
            
            ltv = data['ltv']
            borrow = collateral * ltv
        else:  # mode_borrow
            if 'borrow' not in data:
                await msg.answer("❌ Отсутствует сумма займа\n\nНачните заново с /start")
                await state.clear()
                return
            
            borrow = data['borrow']
            ltv = borrow / collateral if collateral > 0 else 0
        
        ltv_percent = ltv * 100
        
        # Валидация параметров
        if ltv > max_ltv:
            await msg.answer(
                f"❌ Текущий LTV ({ltv_percent:.1f}%) превышает "
                f"Maximum LTV ({max_ltv_input}%)\n\n"
                "Пожалуйста, скорректируйте параметры или начните заново с /start"
            )
            return
        
        if lt <= ltv:
            await msg.answer(
                f"❌ Liquidation Threshold ({lt*100:.1f}%) должен быть больше "
                f"LTV ({ltv_percent:.1f}%)\n\n"
                "Пожалуйста, скорректируйте параметры или начните заново с /start"
            )
            return
        
        # Основные расчеты
        hf = calculate_health_factor(collateral, lt, borrow)
        liquidation_price = calculate_liquidation_price(borrow, supply_amt, lt)
        max_borrow = collateral * max_ltv
        buffer = ((price - liquidation_price) / price) * 100 if price > 0 else 0
        
        # Сценарии падения цены
        scenarios = []
        for drop_percent in [10, 20, 30]:
            new_price = price * (1 - drop_percent / 100)
            new_collateral = supply_amt * new_price
            scenario_hf = calculate_health_factor(new_collateral, lt, borrow)
            scenarios.append((drop_percent, scenario_hf))
        
        # Определяем статус позиции
        status, status_emoji = get_position_status(hf)
        
        # Формируем ответ
        result_message = build_result_message(
            status_emoji, status,
            data.get('supply_ticker', 'N/A'),
            data.get('borrow_ticker', 'N/A'),
            supply_amt, price, collateral,
            borrow, ltv_percent, max_ltv_input, lt,
            hf, liquidation_price, buffer, max_borrow,
            scenarios
        )
        
        # Добавляем рекомендации при необходимости
        if hf < 1.3:
            result_message += (
                "\n\n<b>⚠️ РЕКОМЕНДАЦИИ:</b>\n"
                "• Увеличьте залог для повышения Health Factor\n"
                "• Уменьшите сумму займа\n"
                "• Подготовьте средства для пополнения залога\n"
                "• Установите алерты на изменение цены актива"
            )
        
        await msg.answer(result_message)
        
        # Предлагаем начать новый расчет
        await msg.answer(
            "📝 Для нового расчета используйте /start\n"
            "ℹ️ Для помощи - /help"
        )
        
        await state.clear()
        
    except ZeroDivisionError:
        await msg.answer(
            "❌ Ошибка: деление на ноль. Проверьте введенные данные.\n"
            "Используйте /start для нового расчета."
        )
        await state.clear()
    except Exception as e:
        await msg.answer(
            f"❌ Произошла ошибка: {str(e)}\n\n"
            "Пожалуйста, начните заново с /start"
        )
        await state.clear()

def build_result_message(
    status_emoji: str, status: str,
    supply_ticker: str, borrow_ticker: str,
    supply_amt: float, price: float, collateral: float,
    borrow: float, ltv_percent: float, max_ltv_input: float, lt: float,
    hf: float, liquidation_price: float, buffer: float, max_borrow: float,
    scenarios: list
) -> str:
    """Формирует итоговое сообщение с результатами расчета"""
    
    return (
        f"<b>{status_emoji} РАСЧЕТ ПОЗИЦИИ</b>\n"
        f"Статус: <b>{status}</b>\n\n"
        
        f"<b>💎 ЗАЛОГ:</b>\n"
        f"• Актив: {supply_ticker}\n"
        f"• Количество: {supply_amt:.6f}\n"
        f"• Цена: ${price:.2f}\n"
        f"• Стоимость: <b>{format_currency(collateral)}</b>\n\n"
        
        f"<b>💰 ЗАЙМ:</b>\n"
        f"• Актив: {borrow_ticker}\n"
        f"• Сумма: <b>{format_currency(borrow)}</b>\n\n"
        
        f"<b>⚙️ ПАРАМЕТРЫ:</b>\n"
        f"• Current LTV: <b>{ltv_percent:.2f}%</b>\n"
        f"• Maximum LTV: {max_ltv_input}%\n"
        f"• Liquidation Threshold: {lt*100:.1f}%\n\n"
        
        f"<b>📊 РИСКИ:</b>\n"
        f"• Health Factor: <b>{format_number(hf, 2)}</b>\n"
        f"• Цена ликвидации: <b>${liquidation_price:.2f}</b>\n"
        f"• Буфер безопасности: <b>{buffer:.1f}%</b>\n"
        f"• Макс. возможный займ: {format_currency(max_borrow)}\n\n"
        
        f"<b>📉 СЦЕНАРИИ (падение цены):</b>\n"
        + "\n".join([
            f"• -{drop}% (${price * (1 - drop/100):.2f}) → HF: {format_number(scenario_hf, 2)}"
            for drop, scenario_hf in scenarios
        ])
    )

# ---------- FALLBACK HANDLER ----------
@dp.message()
async def fallback_handler(msg: types.Message, state: FSMContext):
    """Обработчик любых других сообщений"""
    current_state = await state.get_state()
    
    if current_state:
        await msg.answer(
            "⚠️ Пожалуйста, следуйте инструкциям выше.\n"
            "Используйте /reset для отмены текущего расчета."
        )
    else:
        await msg.answer(
            "👋 Привет! Я помогу рассчитать параметры вашей DeFi позиции.\n\n"
            "Используйте /start для начала расчета\n"
                    )

# ---------- ERROR HANDLING ----------
@dp.error()
async def error_handler(event, exception):
    """Глобальный обработчик ошибок"""
    print(f"❌ Ошибка: {exception}")
    return True

# ---------- RUN ----------
async def main():
    """Основная функция запуска бота"""
    print("=" * 50)
    print("🚀 DeFi Position Calculator Bot")
    print("=" * 50)
    print("✅ Бот успешно запущен")
    print("ℹ️  Нажмите Ctrl+C для остановки")
    print("=" * 50)
    
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("👋 Бот остановлен пользователем")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")

