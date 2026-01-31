import asyncio
import os
from aiogram.client.default import DefaultBotProperties
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.strategy import FSMStrategy
from decimal import Decimal, ROUND_DOWN

# Получаем токен из переменных окружения
TOKEN = ("BOT_TOKEN")

if not TOKEN:
    raise ValueError(
        "Не установлен токен бота. Установите переменную окружения BOT_TOKEN")

bot = Bot(
    token=TOKEN,
    default=DefaultBotProperties(parse_mode="HTML")
)
dp = Dispatcher(storage=MemoryStorage(), fsm_strategy=FSMStrategy.USER_IN_CHAT)

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

# ---------- KEYBOARD ----------
mode_kb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🔢 По LTV", callback_data="mode_ltv")],
    [InlineKeyboardButton(text="💵 По сумме займа", callback_data="mode_borrow")]
])

# ---------- VALIDATION HELPERS ----------
def validate_number(text: str, min_val: float = 0, max_val: float = None) -> tuple[bool, float, str]:
    """Проверяет корректность числового ввода"""
    try:
        # Заменяем запятые на точки для корректного парсинга
        text = text.replace(",", ".").strip()
        value = float(text)
        
        if value <= min_val:
            return False, 0, f"Значение должно быть больше {min_val}"
        if max_val is not None and value > max_val:
            return False, 0, f"Значение должно быть не больше {max_val}"
        
        return True, value, ""
    except (ValueError, TypeError):
        return False, 0, "Пожалуйста, введите корректное число"

def format_currency(value: float) -> str:
    """Форматирует денежные значения"""
    if value >= 1000000:
        return f"${value/1000000:.2f}M"
    elif value >= 1000:
        return f"${value/1000:.1f}K"
    else:
        return f"${value:.2f}"

# ---------- START ----------
@dp.message(Command("start"))
async def start_cmd(msg: types.Message, state: FSMContext):
    """Обработчик команды /start"""
    await state.clear()
    await msg.answer(
        "<b>Калькулятор позиции DeFi</b>\n\n"
        "Введите тикер залогового актива (например: ETH, SOL, BTC):"
    )
    await state.set_state(Calc.supply_ticker)

@dp.message(Command("reset", "отмена", "сброс"))
async def reset_cmd(msg: types.Message, state: FSMContext):
    """Обработчик команды сброса"""
    await state.clear()
    await msg.answer("✅ Состояние сброшено. Используйте /start для начала расчета.")

@dp.message(Command("help", "помощь"))
async def help_cmd(msg: types.Message):
    """Обработчик команды помощи"""
    await msg.answer(
        "<b>Помощь по боту:</b>\n\n"
        "/start - начать расчет позиции\n"
        "/reset - сбросить текущий расчет\n"
        "/help - показать это сообщение\n\n"
        "Бот рассчитывает параметры кредитной позиции в DeFi:\n"
        "• Health Factor\n"
        "• Цену ликвидации\n"
        "• Максимальный займ\n"
        "• Буфер безопасности"
    )

# ---------- FLOW ----------
@dp.message(Calc.supply_ticker)
async def process_supply_ticker(msg: types.Message, state: FSMContext):
    """Обработка тикера залогового актива"""
    ticker = msg.text.upper().strip()
    if len(ticker) > 10:
        await msg.answer("Слишком длинный тикер. Введите корректный тикер (например: ETH):")
        return
    
    await state.update_data(supply_ticker=ticker)
    await msg.answer(
        f"Залоговый актив: <b>{ticker}</b>\n\n"
        "Введите тикер заимствуемого актива (например: USDC, DAI, USDT):"
    )
    await state.set_state(Calc.borrow_ticker)

@dp.message(Calc.borrow_ticker)
async def process_borrow_ticker(msg: types.Message, state: FSMContext):
    """Обработка тикера заимствуемого актива"""
    ticker = msg.text.upper().strip()
    if len(ticker) > 10:
        await msg.answer("Слишком длинный тикер. Введите корректный тикер:")
        return
    
    await state.update_data(borrow_ticker=ticker)
    await msg.answer(
        f"Заимствуемый актив: <b>{ticker}</b>\n\n"
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
        f"Залоговый актив: <b>{data.get('supply_ticker')}</b>\n"
        f"Количество: <b>{value:.6f}</b>\n\n"
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
        f"<b>📊 Предварительный расчет:</b>\n\n"
        f"Залоговый актив: {data.get('supply_ticker')}\n"
        f"Количество: {supply_amount:.6f}\n"
        f"Цена: ${value:.2f}\n"
        f"<b>Стоимость залога: ${collateral_value:.2f}</b>\n\n"
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
            f"<b>Режим: Расчет по LTV</b>\n\n"
            f"Стоимость залога: ${collateral_value:.2f}\n\n"
            "Введите Loan-to-Value (LTV) в % (например: 50):"
        )
        await state.set_state(Calc.ltv)
    else:  # mode_borrow
        await cb.message.edit_text(
            f"<b>Режим: Расчет по сумме займа</b>\n\n"
            f"Стоимость залога: ${collateral_value:.2f}\n\n"
            "Введите сумму займа:"
        )
        await state.set_state(Calc.borrow)

@dp.message(Calc.ltv)
async def process_ltv(msg: types.Message, state: FSMContext):
    """Обработка LTV"""
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    
    if not valid:
        await msg.answer(f"❌ {error}\n\nLTV должен быть от 0 до 100%.\nВведите LTV (%):")
        return
    
    await state.update_data(ltv=value / 100)
    
    data = await state.get_data()
    supply_amount = data.get('supply_amount', 0)
    supply_price = data.get('supply_price', 0)
    collateral_value = supply_amount * supply_price
    borrow_amount = collateral_value * (value / 100)
    
    await msg.answer(
        f"<b>LTV: {value}%</b>\n"
        f"Сумма займа при таком LTV: ${borrow_amount:.2f}\n\n"
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
            f"❌ Сумма займа (${value:.2f}) превышает стоимость залога (${collateral_value:.2f})\n\n"
            "Введите сумму займа:"
        )
        return
    
    await state.update_data(borrow=value)
    
    ltv_percent = (value / collateral_value) * 100 if collateral_value > 0 else 0
    
    await msg.answer(
        f"<b>Сумма займа: ${value:.2f}</b>\n"
        f"LTV при такой сумме: {ltv_percent:.1f}%\n\n"
        "Введите Liquidation Threshold (LT) в % (например: 75):"
    )
    await state.set_state(Calc.lt)

@dp.message(Calc.lt)
async def process_lt(msg: types.Message, state: FSMContext):
    """Обработка Liquidation Threshold"""
    valid, value, error = validate_number(msg.text, min_val=0, max_val=100)
    
    if not valid:
        await msg.answer(f"❌ {error}\n\nLT должен быть от 0 до 100%.\nВведите LT (%):")
        return
    
    await state.update_data(lt=value / 100)
    
    await msg.answer(
        f"<b>Liquidation Threshold: {value}%</b>\n\n"
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
        required_fields = ['supply_amount', 'supply_price', 'lt', 'mode']
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
            ltv_percent = ltv * 100
        else:  # mode_borrow
            if 'borrow' not in data:
                await msg.answer("❌ Отсутствует сумма займа\n\nНачните заново с /start")
                await state.clear()
                return
            
            borrow = data['borrow']
            ltv = borrow / collateral if collateral > 0 else 0
            ltv_percent = ltv * 100
        
        # Проверяем, что LTV не превышает Max LTV
        if ltv > max_ltv:
            await msg.answer(
                f"❌ Текущий LTV ({ltv_percent:.1f}%) превышает Maximum LTV ({max_ltv_input}%)\n\n"
                "Начните заново с /start"
            )
            await state.clear()
            return
        
        # Проверяем, что LT больше LTV
        if lt <= ltv:
            await msg.answer(
                f"❌ Liquidation Threshold ({lt*100:.1f}%) должен быть больше LTV ({ltv_percent:.1f}%)\n\n"
                "Начните заново с /start"
            )
            await state.clear()
            return
        
        # Основные расчеты
        # Health Factor
        hf = (collateral * lt) / borrow if borrow > 0 else float('inf')
        
        # Цена ликвидации
        liquidation_price = borrow / (supply_amt * lt) if (supply_amt * lt) > 0 else 0
        
        # Максимальный займ
        max_borrow = collateral * max_ltv
        
        # Буфер безопасности
        buffer = ((price - liquidation_price) / price) * 100 if price > 0 else 0
        
        # Сценарии
        price_10 = price * 0.9
        price_20 = price * 0.8
        price_30 = price * 0.7
        
        if borrow > 0:
            hf_10 = (supply_amt * price_10 * lt) / borrow
            hf_20 = (supply_amt * price_20 * lt) / borrow
            hf_30 = (supply_amt * price_30 * lt) / borrow
        else:
            hf_10 = hf_20 = hf_30 = float('inf')
        
        # Определяем статус позиции
        if hf <= 1.0:
            status = "🔴 ЛИКВИДАЦИЯ"
            status_emoji = "🔴"
        elif hf < 1.3:
            status = "🟡 ВНИМАНИЕ"
            status_emoji = "🟡"
        elif hf < 2.0:
            status = "🟢 БЕЗОПАСНО"
            status_emoji = "🟢"
        else:
            status = "🔵 ОЧЕНЬ БЕЗОПАСНО"
            status_emoji = "🔵"
        
        # Формируем ответ
        result_message = (
            f"<b>{status_emoji} РАСЧЕТ ПОЗИЦИИ</b>\n\n"
            
            f"<b>Залог:</b>\n"
            f"• Актива: {data.get('supply_ticker', 'N/A')}\n"
            f"• Количество: {supply_amt:.6f}\n"
            f"• Цена: ${price:.2f}\n"
            f"• Стоимость: <b>${collateral:.2f}</b>\n\n"
            
            f"<b>Займ:</b>\n"
            f"• Актива: {data.get('borrow_ticker', 'N/A')}\n"
            f"• Сумма: <b>${borrow:.2f}</b>\n\n"
            
            f"<b>Параметры:</b>\n"
            f"• LTV: <b>{ltv_percent:.2f}%</b>\n"
            f"• Max LTV: {max_ltv_input}%\n"
            f"• Liquidation Threshold: {lt*100:.1f}%\n\n"
            
            f"<b>Риски:</b>\n"
            f"• Health Factor: <b>{hf:.2f}</b> ({status})\n"
            f"• Цена ликвидации: <b>${liquidation_price:.2f}</b>\n"
            f"• Буфер: <b>{buffer:.1f}%</b>\n"
            f"• Max займ: {format_currency(max_borrow)}\n\n"
            
            f"<b>Сценарии:</b>\n"
            f"• -10% цена → HF: {hf_10:.2f}\n"
            f"• -20% цена → HF: {hf_20:.2f}\n"
            f"• -30% цена → HF: {hf_30:.2f}"
        )
        
        # Добавляем рекомендации
        if hf < 1.3:
            recommendations = (
                "\n\n<b>⚠️ РЕКОМЕНДАЦИИ:</b>\n"
                "1. Увеличьте залог\n"
                "2. Уменьшите сумму займа\n"
                "3. Будьте готовы к пополнению залога"
            )
            result_message += recommendations
        
        await msg.answer(result_message)
        
        # Предлагаем начать новый расчет
        await msg.answer(
            "Для нового расчета используйте /start\n"
            "Для помощи - /help"
        )
        
        await state.clear()
        
    except ZeroDivisionError:
        await msg.answer("❌ Ошибка: деление на ноль. Проверьте введенные данные.")
        await state.clear()
    except Exception as e:
        await msg.answer(f"❌ Произошла ошибка: {str(e)}\n\nПожалуйста, начните заново с /start")
        await state.clear()

# ---------- FALLBACK HANDLER ----------
@dp.message()
async def fallback_handler(msg: types.Message, state: FSMContext):
    """Обработчик любых других сообщений"""
    current_state = await state.get_state()
    
    if current_state:
        await msg.answer(
            "⚠️ Пожалуйста, следуйте инструкциям выше.\n"
            "Или используйте /reset для отмены текущего расчета."
        )
    else:
        await msg.answer(
            "Для начала расчета позиции используйте /start\n"
            "Для помощи - /help"
        )

# ---------- RUN ----------
async def main():
    """Основная функция запуска бота"""
    print("🚀 Бот запущен...")
    print("ℹ️  Используйте Ctrl+C для остановки")
    
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен")

