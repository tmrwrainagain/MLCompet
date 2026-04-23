"""Telegram bot for Module G.

Integrates with the FastAPI via httpx.
Commands: /start /assess /parallel /sequential /complexity /time /trajectory /cancel
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict

import httpx
from telegram import (
    InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove, Update,
)
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler,
    ContextTypes, ConversationHandler, MessageHandler, filters,
)

from .config import API_BASE_URL, TELEGRAM_BOT_TOKEN

# ── Conversation states ───────────────────────────────────────────────────────
(
    ASSESS_MODE, ASSESS_ID, ASSESS_TEXT,
    CLUSTER_INPUT,
    COMPLEXITY_INPUT,
    TIME_SCOPE, TIME_MATERIAL, TIME_SUBJECT, TIME_SUBJECTS, TIME_PARALLEL,
    TRAJ_SUBJECT, TRAJ_DIFF, TRAJ_HOURS, TRAJ_STYLE, TRAJ_TOPICS,
) = range(15)


# ── HTTP helper ───────────────────────────────────────────────────────────────

async def _post(endpoint: str, payload: Dict[str, Any]) -> Dict | None:
    url = f"{API_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        return {"error": f"Ошибка API {e.response.status_code}: {detail}"}
    except Exception as e:
        return {"error": f"API недоступен: {e}"}


def _fmt_minutes(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h} ч {m} мин" if h > 0 else f"{m} мин"


def _chunk(text: str, size: int = 4000):
    return textwrap.wrap(text, size, break_long_words=False, replace_whitespace=False)


async def _send(update: Update, text: str):
    for part in _chunk(text):
        if update.message:
            await update.message.reply_text(part, parse_mode="HTML")
        elif update.callback_query:
            await update.callback_query.message.reply_text(part, parse_mode="HTML")


# ── /start ────────────────────────────────────────────────────────────────────

MAIN_MENU = InlineKeyboardMarkup([
    [InlineKeyboardButton("Оценка материала",         callback_data="cmd_assess")],
    [InlineKeyboardButton("Параллельное изучение",    callback_data="cmd_parallel"),
     InlineKeyboardButton("Последовательное",         callback_data="cmd_sequential")],
    [InlineKeyboardButton("Уровень сложности",        callback_data="cmd_complexity")],
    [InlineKeyboardButton("Время освоения",           callback_data="cmd_time")],
    [InlineKeyboardButton("Траектория обучения",      callback_data="cmd_trajectory")],
])

WELCOME = (
    "<b>Система анализа учебных материалов</b>\n\n"
    "Я помогу вам:\n"
    "• оценить материал по методическим критериям\n"
    "• определить возможность параллельного или последовательного изучения\n"
    "• установить уровень сложности\n"
    "• рассчитать время освоения\n"
    "• построить индивидуальную траекторию\n\n"
    "Выберите действие:"
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME, parse_mode="HTML", reply_markup=MAIN_MENU)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "<b>Доступные команды:</b>\n"
        "/assess — оценить учебный материал\n"
        "/parallel — группы параллельного изучения\n"
        "/sequential — порядок последовательного изучения\n"
        "/complexity — уровень сложности\n"
        "/time — время освоения материала/предмета\n"
        "/trajectory — индивидуальная траектория\n"
        "/cancel — отменить текущий диалог\n"
        "/start — главное меню"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("Диалог отменён.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


# ── /assess conversation ──────────────────────────────────────────────────────

async def assess_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("Ввести текст", callback_data="assess_text"),
        InlineKeyboardButton("ID материала", callback_data="assess_id"),
    ]])
    msg = update.message or update.callback_query.message
    await msg.reply_text("Как хотите передать материал?", reply_markup=kb)
    return ASSESS_MODE


async def assess_mode_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    choice = update.callback_query.data
    if choice == "assess_id":
        await update.callback_query.message.reply_text("Введите ID материала (число):")
        return ASSESS_ID
    await update.callback_query.message.reply_text("Отправьте текст материала:")
    return ASSESS_TEXT


async def assess_by_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        mid = int(update.message.text.strip())
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите числовой ID.")
        return ASSESS_ID

    await update.message.reply_text("Анализирую...")
    data = await _post("assess", {"material_id": mid})
    await _send(update, _format_assess(data))
    return ConversationHandler.END


async def assess_by_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    await update.message.reply_text("Анализирую... (может занять до 30 с)")
    data = await _post("assess", {"text": text})
    await _send(update, _format_assess(data))
    return ConversationHandler.END


def _format_assess(data: dict) -> str:
    if "error" in data:
        return f"Ошибка: {data['error']}"
    score  = data.get("overall_score", 0)
    status = data.get("moderation_status", "")
    status_emoji = {"approved": "✅", "rejected": "❌", "needs_revision": "⚠️"}.get(status, "ℹ️")

    cats: dict[str, list] = {}
    for r in data.get("requirements", []):
        cats.setdefault(r["category"], []).append(r)

    cats_text = ""
    for cat, reqs in cats.items():
        met   = sum(1 for r in reqs if r["is_met"])
        total = len(reqs)
        cats_text += f"\n  <b>{cat}</b>: {met}/{total}\n"
        for r in reqs:
            icon = "✔" if r["is_met"] else "✘"
            cats_text += f"    {icon} {r['requirement']} ({r['score']:.1f})\n"

    strengths = "\n".join(f"• {s}" for s in data.get("strengths", []))
    weaknesses = "\n".join(f"• {w}" for w in data.get("weaknesses", []))

    return (
        f"{status_emoji} <b>Оценка материала</b>\n"
        f"Предмет: {data.get('subject') or '—'}  |  Тема: {data.get('topic') or '—'}\n"
        f"Итоговый балл: <b>{score:.1f}/10</b>\n"
        f"Статус: {status}\n"
        f"{'(из кэша БД)' if data.get('cached') else ''}\n\n"
        f"<b>Требования по категориям:</b>{cats_text}\n"
        f"<b>Достоинства:</b>\n{strengths or '—'}\n\n"
        f"<b>Недостатки:</b>\n{weaknesses or '—'}\n\n"
        f"<b>Рекомендации:</b>\n{data.get('recommendations') or '—'}"
    )


# ── /parallel and /sequential conversations ───────────────────────────────────

async def cluster_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    cmd = update.message.text.split()[0] if update.message else "/parallel"
    context.user_data["cluster_cmd"] = "parallel" if "parallel" in cmd else "sequential"
    label = "параллельного" if "parallel" in cmd else "последовательного"
    await update.message.reply_text(
        f"Отправьте тексты материалов для анализа {label} изучения.\n"
        "Разделяйте материалы тремя дефисами: <code>---</code>",
        parse_mode="HTML",
    )
    return CLUSTER_INPUT


async def cluster_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    raw   = update.message.text.strip()
    parts = [p.strip() for p in raw.split("---") if p.strip()]
    if len(parts) < 2:
        await update.message.reply_text("Нужно минимум 2 материала, разделённых <code>---</code>.", parse_mode="HTML")
        return CLUSTER_INPUT

    mats    = [{"text": p, "title": f"Материал {i+1}"} for i, p in enumerate(parts)]
    cmd     = context.user_data.get("cluster_cmd", "parallel")
    endpoint = "parallel" if cmd == "parallel" else "sequential"

    await update.message.reply_text("Анализирую...")
    data = await _post(endpoint, {"materials": mats})

    if "error" in data:
        await _send(update, f"Ошибка: {data['error']}")
        return ConversationHandler.END

    if cmd == "parallel":
        text = f"<b>Параллельное изучение — {data['total_groups']} групп</b>\n\n"
        for g in data.get("groups", []):
            text += f"<b>Группа {g['group_id']+1}: {g['label']}</b>\n"
            for t in g["titles"]:
                text += f"  • {t}\n"
            text += "\n"
        text += f"💡 {data.get('recommendation', '')}"
    else:
        text = "<b>Последовательный порядок изучения:</b>\n\n"
        for item in data.get("order", []):
            text += f"{item['position']}. <b>{item['title']}</b>\n"
            text += f"   Фаза: {item['phase_name']}\n"
            if item.get("subject"):
                text += f"   Предмет: {item['subject']}\n"
            text += "\n"
        text += f"\n{data.get('explanation', '')}"

    await _send(update, text)
    return ConversationHandler.END


# ── /complexity conversation ──────────────────────────────────────────────────

async def complexity_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Отправьте текст материала для определения уровня сложности:")
    return COMPLEXITY_INPUT


async def complexity_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    await update.message.reply_text("Определяю уровень сложности...")
    data = await _post("complexity", {"text": text})

    if "error" in data:
        await _send(update, f"Ошибка: {data['error']}")
        return ConversationHandler.END

    level_emoji = {"Базовый": "🟢", "Средний": "🟡", "Продвинутый": "🔴"}.get(data.get("level", ""), "⚪")
    result = (
        f"{level_emoji} <b>Уровень сложности: {data.get('level')}</b>\n\n"
        f"Балл: {data.get('score', 0):.2f}\n"
        f"Слов в тексте: {data.get('word_count', 0)}\n"
        f"Средняя длина предложения: {data.get('avg_sentence_length', 0)} слов\n\n"
        f"{data.get('explanation', '')}"
    )
    await _send(update, result)
    return ConversationHandler.END


# ── /time conversation ────────────────────────────────────────────────────────

async def time_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Один материал (текст)", callback_data="time_material")],
        [InlineKeyboardButton("Предмет (из БД)",       callback_data="time_subject")],
        [InlineKeyboardButton("Набор предметов",        callback_data="time_subjects")],
    ])
    msg = update.message or update.callback_query.message
    await msg.reply_text("Что хотите оценить?", reply_markup=kb)
    return TIME_SCOPE


async def time_scope_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    scope = update.callback_query.data
    context.user_data["time_scope"] = scope

    if scope == "time_material":
        await update.callback_query.message.reply_text("Отправьте текст материала:")
        return TIME_MATERIAL
    elif scope == "time_subject":
        await update.callback_query.message.reply_text("Введите название предмета:")
        return TIME_SUBJECT
    else:
        await update.callback_query.message.reply_text("Введите предметы через запятую:")
        return TIME_SUBJECTS


async def time_material_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    await update.message.reply_text("Считаю время...")
    data = await _post("time/material", {"text": text})

    if "error" in data:
        await _send(update, f"Ошибка: {data['error']}")
        return ConversationHandler.END

    bd = data.get("breakdown", {})
    result = (
        f"⏱ <b>Оценка времени освоения</b>\n\n"
        f"Тип занятия: {data.get('lesson_type', '—')}\n"
        f"Уровень сложности: {data.get('difficulty_label', '—')}\n"
        f"Слов в тексте: {data.get('word_count', 0)}\n\n"
        f"<b>Итого: {data.get('human_readable', '—')}</b>\n\n"
        f"Разбивка:\n"
        f"• Чтение: {bd.get('reading_minutes', 0):.0f} мин\n"
        f"• Надбавка за сложность: {bd.get('complexity_overhead_minutes', 0):.0f} мин\n"
        f"• Практика: {bd.get('practice_minutes', 0)} мин\n"
        f"• Видео: {bd.get('video_minutes', 0)} мин"
    )
    await _send(update, result)
    return ConversationHandler.END


async def time_subject_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    subject = update.message.text.strip()
    await update.message.reply_text(f"Считаю время для предмета «{subject}»...")
    data = await _post("time/subject", {"subject": subject})

    if "error" in data:
        await _send(update, f"Ошибка: {data['error']}")
        return ConversationHandler.END

    by_lt   = "\n".join(f"  • {k}: {_fmt_minutes(v)}" for k, v in data.get("by_lesson_type", {}).items())
    by_diff = "\n".join(f"  • {k}: {_fmt_minutes(v)}" for k, v in data.get("by_difficulty", {}).items())

    result = (
        f"⏱ <b>Время освоения: {data.get('subject')}</b>\n\n"
        f"Материалов: {data.get('material_count', 0)}\n"
        f"<b>Итого: {data.get('human_readable', '—')}</b>\n\n"
        f"По типу занятий:\n{by_lt or '  —'}\n\n"
        f"По сложности:\n{by_diff or '  —'}"
    )
    await _send(update, result)
    return ConversationHandler.END


async def time_subjects_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    subjects = [s.strip() for s in update.message.text.split(",") if s.strip()]
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("Последовательно", callback_data="ts_seq"),
        InlineKeyboardButton("Параллельно",     callback_data="ts_par"),
    ]])
    context.user_data["time_subjects"] = subjects
    await update.message.reply_text("Как планируете изучать?", reply_markup=kb)
    return TIME_PARALLEL


async def time_parallel_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    parallel = update.callback_query.data == "ts_par"
    subjects = context.user_data.get("time_subjects", [])

    await update.callback_query.message.reply_text("Считаю...")
    data = await _post("time/subjects", {"subjects": subjects, "parallel_study": parallel})

    if "error" in data:
        await _send(update, f"Ошибка: {data['error']}")
        return ConversationHandler.END

    lines = "\n".join(
        f"• {s['subject']}: {_fmt_minutes(s['total_minutes'])} ({s.get('material_count',0)} мат.)"
        for s in data.get("subjects", [])
    )
    result = (
        f"⏱ <b>Время освоения набора предметов</b>\n\n"
        f"{lines}\n\n"
        f"Последовательно: <b>{data.get('human_readable_sequential', '—')}</b>\n"
        f"Параллельно: <b>{data.get('human_readable_parallel', '—')}</b>"
    )
    await _send(update, result)
    return ConversationHandler.END


# ── /trajectory conversation ──────────────────────────────────────────────────

async def traj_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    msg = update.message or update.callback_query.message
    await msg.reply_text("Введите название предмета для построения траектории:")
    return TRAJ_SUBJECT


async def traj_subject(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["traj_subject"] = update.message.text.strip()
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("Базовый",      callback_data="diff_Базовый"),
        InlineKeyboardButton("Средний",      callback_data="diff_Средний"),
        InlineKeyboardButton("Продвинутый",  callback_data="diff_Продвинутый"),
    ]])
    await update.message.reply_text("Выберите целевой уровень сложности:", reply_markup=kb)
    return TRAJ_DIFF


async def traj_diff_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    context.user_data["traj_diff"] = update.callback_query.data.replace("diff_", "")
    await update.callback_query.message.reply_text("Сколько часов в неделю вы готовы посвящать учёбе? (например: 5)")
    return TRAJ_HOURS


async def traj_hours(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        hours = float(update.message.text.strip().replace(",", "."))
        if hours <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("Введите положительное число (например: 5.5)")
        return TRAJ_HOURS
    context.user_data["traj_hours"] = hours

    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("Последовательное", callback_data="style_sequential"),
        InlineKeyboardButton("Смешанное",        callback_data="style_mixed"),
    ]])
    await update.message.reply_text("Стиль обучения:", reply_markup=kb)
    return TRAJ_STYLE


async def traj_style_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    context.user_data["traj_style"] = update.callback_query.data.replace("style_", "")
    await update.callback_query.message.reply_text(
        "Введите темы, которые хотите изучить (через запятую), или /skip чтобы включить все:"
    )
    return TRAJ_TOPICS


async def traj_topics(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    topics = [] if text.lower() in ("/skip", "skip", "все") else [t.strip() for t in text.split(",") if t.strip()]
    context.user_data["traj_topics"] = topics
    return await _build_trajectory(update, context)


async def _build_trajectory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    ud = context.user_data
    payload = {
        "subject":                  ud.get("traj_subject"),
        "difficulty_level":         ud.get("traj_diff", "Средний"),
        "available_hours_per_week": ud.get("traj_hours", 5),
        "learning_style":           ud.get("traj_style", "sequential"),
        "target_topics":            ud.get("traj_topics", []),
    }
    await update.message.reply_text("Строю траекторию...")
    data = await _post("trajectory", payload)

    if "error" in data:
        await _send(update, f"Ошибка: {data['error']}")
        return ConversationHandler.END

    steps = data.get("trajectory", [])
    total_min = data.get("total_estimated_minutes", 0)
    weeks = data.get("total_weeks", 0)

    header = (
        f"🗺 <b>Индивидуальная траектория — {payload['subject']}</b>\n"
        f"Уровень: {payload['difficulty_level']} | Стиль: {payload['learning_style']}\n"
        f"Материалов: {data.get('total_steps', 0)} | "
        f"Время: {_fmt_minutes(total_min)} | Недель: {weeks}\n\n"
    )

    plan_lines = ""
    for step in steps:
        plan_lines += (
            f"<b>Шаг {step['step']} (неделя {step['week']})</b>\n"
            f"  {step.get('topic') or f'Материал #{step[\"material_id\"]}'}\n"
            f"  Сложность: {step.get('difficulty_label', '—')} | "
            f"Время: {_fmt_minutes(step['estimated_minutes'])}\n"
            f"  {step.get('rationale', '')}\n\n"
        )

    await _send(update, header + plan_lines)
    context.user_data.clear()
    return ConversationHandler.END


# ── Inline menu callbacks ─────────────────────────────────────────────────────

async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    data = update.callback_query.data

    if data == "cmd_assess":
        return await assess_start.__wrapped__(update, context) if hasattr(assess_start, "__wrapped__") else await assess_start(update, context)
    if data == "cmd_time":
        return await time_start(update, context)
    if data == "cmd_trajectory":
        return await traj_start(update, context)

    cmd_map = {
        "cmd_parallel":    "/parallel",
        "cmd_sequential":  "/sequential",
        "cmd_complexity":  "/complexity",
    }
    if data in cmd_map:
        msg = update.callback_query.message
        msg.text = cmd_map[data]
        if data == "cmd_complexity":
            return await complexity_start(update, context)
        return await cluster_start(update, context)


# ── Application factory ───────────────────────────────────────────────────────

def build_app() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # /start, /help
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help",  help_cmd))

    # /assess
    assess_conv = ConversationHandler(
        entry_points=[CommandHandler("assess", assess_start)],
        states={
            ASSESS_MODE: [CallbackQueryHandler(assess_mode_cb)],
            ASSESS_ID:   [MessageHandler(filters.TEXT & ~filters.COMMAND, assess_by_id)],
            ASSESS_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, assess_by_text)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(assess_conv)

    # /parallel, /sequential
    cluster_conv = ConversationHandler(
        entry_points=[
            CommandHandler("parallel",   cluster_start),
            CommandHandler("sequential", cluster_start),
        ],
        states={
            CLUSTER_INPUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, cluster_input)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(cluster_conv)

    # /complexity
    complexity_conv = ConversationHandler(
        entry_points=[CommandHandler("complexity", complexity_start)],
        states={
            COMPLEXITY_INPUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, complexity_input)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(complexity_conv)

    # /time
    time_conv = ConversationHandler(
        entry_points=[CommandHandler("time", time_start)],
        states={
            TIME_SCOPE:    [CallbackQueryHandler(time_scope_cb)],
            TIME_MATERIAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, time_material_input)],
            TIME_SUBJECT:  [MessageHandler(filters.TEXT & ~filters.COMMAND, time_subject_input)],
            TIME_SUBJECTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, time_subjects_input)],
            TIME_PARALLEL: [CallbackQueryHandler(time_parallel_cb)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(time_conv)

    # /trajectory
    traj_conv = ConversationHandler(
        entry_points=[CommandHandler("trajectory", traj_start)],
        states={
            TRAJ_SUBJECT: [MessageHandler(filters.TEXT & ~filters.COMMAND, traj_subject)],
            TRAJ_DIFF:    [CallbackQueryHandler(traj_diff_cb)],
            TRAJ_HOURS:   [MessageHandler(filters.TEXT & ~filters.COMMAND, traj_hours)],
            TRAJ_STYLE:   [CallbackQueryHandler(traj_style_cb)],
            TRAJ_TOPICS:  [MessageHandler(filters.TEXT & ~filters.COMMAND, traj_topics)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(traj_conv)

    # Main menu inline buttons
    app.add_handler(CallbackQueryHandler(menu_callback, pattern="^cmd_"))

    return app


def run():
    print(f"  Starting Telegram bot (API: {API_BASE_URL})")
    build_app().run_polling(drop_pending_updates=True)
