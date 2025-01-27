import asyncio
import os
import subprocess
import time
import logging
import sys
import urllib.parse
import re
import sqlite3
from datetime import datetime, timedelta, timezone
import aiohttp
import json
import xml.etree.ElementTree as ET
import m3u8
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from typing import Any, List, Set, Dict, Tuple, Union

# --------------------------------------------------------------------------
# Настройка логирования
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  # Для отладки можно изменить на DEBUG
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------

def clean_url(url: str) -> str:
    """
    Нормализует URL: удаляет фрагменты и параметры запроса, приводит к нижнему регистру и т.д.
    """
    parsed = urllib.parse.urlparse(url)
    normalized = parsed._replace(fragment='', query='').geturl()
    return normalized.strip().lower()

def extract_nickname(url: str) -> str:
    """
    Извлекает никнейм из URL (последний сегмент пути).
    """
    parsed = urllib.parse.urlparse(url)
    path_segments = parsed.path.strip('/').split('/')
    if path_segments:
        return path_segments[-1]
    return "unknown"

def parse_domain(url: str) -> str:
    """
    Выделяет домен (site_name) из заданного URL.
    """
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc or "unknown_site"

def is_stream_url(url: str, ctype: str = "") -> bool:
    """
    Определяет, является ли URL потенциальной стрим-ссылкой.
    Проверяет наличие определенных расширений в URL и по content-type.
    """
    if not url:
        return False

    url_lower = url.lower()

    # Исключение плейсхолдерных ссылок
    if "autoplayinline" in url_lower:
        return False

    # Определённые расширения для стримов
    stream_extensions = [".m3u8", ".mpd", ".ts", ".flv", ".f4m", ".f4v"]
    if any(url_lower.endswith(ext) for ext in stream_extensions):
        return True

    # Проверка по content-type
    ctype = ctype.lower()
    stream_content_types = [
        "application/vnd.apple.mpegurl",
        "application/x-mpegurl",
        "application/dash+xml",
        "application/vnd.ms-sstr+xml",
        "application/x-f4m",
        "video/mp2t",
        "video/mp4",  # осторожнее с mp4 — может быть просто файлом
        "video/x-flv",
    ]
    if any(ctype.startswith(ct) for ct in stream_content_types):
        return True

    return False

def is_placeholder_link(url: str) -> bool:
    """
    Проверяет, является ли ссылка плейсхолдером (например, 'javascript:void(0)')
    """
    if not url:
        return True
    url_lower = url.lower()
    if url_lower.startswith("javascript:"):
        return True
    if url_lower.startswith("data:"):
        return True
    return False

def decode_js_escapes(s: str) -> str:
    """
    Декодирует JavaScript escape последовательности в строке.
    """
    try:
        return s.encode("utf-8", errors="ignore").decode("unicode_escape", errors="ignore")
    except UnicodeDecodeError:
        return s

# Поиск URL в тексте
URL_PATTERN = re.compile(
    r'''(?i)\b((?:https?://|//)
    [a-z0-9\-._~:/?#[\]@!$&'()*+,;=%]+)''',
    re.VERBOSE
)

def deep_search_for_urls(obj: Any) -> List[str]:
    """
    Рекурсивный поиск URL в словаре/списке/строке (JSON).
    """
    results = []
    if isinstance(obj, dict):
        for val in obj.values():
            results.extend(deep_search_for_urls(val))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(deep_search_for_urls(item))
    elif isinstance(obj, str):
        found = URL_PATTERN.findall(obj)
        results.extend(found)
    return results

def resolve_url(base: str, link: str) -> str:
    """ Преобразует относительный URL в абсолютный. """
    return urllib.parse.urljoin(base, link)

# --------------------------------------------------------------------------
# Парсеры плейлистов
# --------------------------------------------------------------------------
import xml.etree.ElementTree as ET
import m3u8

class HLSPlaylistParser:
    """
    Парсер для HLS (.m3u8) плейлистов.
    """
    def __init__(self, content: str, base_url: str):
        self.playlist = m3u8.M3U8(content, base_uri=base_url)
        self.base_url = base_url

    def parse(self) -> Dict[str, Any]:
        info = {
            "type": "HLS",
            "base_url": self.base_url,
            "version": self.playlist.version,
            "target_duration": self.playlist.target_duration,
            "media_sequence": self.playlist.media_sequence,
            "segments": [],
            "playlists": []
        }

        # Сегменты
        for segment in self.playlist.segments:
            segment_uri = resolve_url(self.base_url, segment.uri)
            info["segments"].append({
                "uri": segment_uri,
                "duration": segment.duration,
                "title": segment.title,
                "byterange": segment.byterange
            })

        # Мастер-плейлисты (варианты качества)
        for pl in self.playlist.playlists:
            stream_info = pl.stream_info
            pl_uri = resolve_url(self.base_url, pl.uri)
            resolution = stream_info.resolution
            quality = f"{resolution[1]}p" if resolution else "unknown"
            info["playlists"].append({
                "uri": pl_uri,
                "bandwidth": stream_info.bandwidth,
                "average_bandwidth": stream_info.average_bandwidth,
                "codecs": stream_info.codecs,
                "resolution": f"{resolution[0]}x{resolution[1]}" if resolution else None,
                "frame_rate": stream_info.frame_rate,
                "hdcp_level": stream_info.hdcp_level,
                "quality": quality
            })
        return info

class DASHPlaylistParser:
    """
    Парсер для DASH (.mpd) плейлистов.
    """
    def __init__(self, content: str, base_url: str):
        self.tree = ET.ElementTree(ET.fromstring(content))
        self.root = self.tree.getroot()
        self.base_url = base_url
        self.ns = self._get_namespaces()

    def _get_namespaces(self) -> Dict[str, str]:
        namespaces = {}
        for ns in self.root.attrib:
            if ns.startswith("xmlns:"):
                prefix = ns.split(":")[1]
                namespaces[prefix] = self.root.attrib[ns]
            elif ns == "xmlns":
                namespaces['default'] = self.root.attrib[ns]
        return namespaces

    def parse(self) -> Dict[str, Any]:
        info = {
            "type": "DASH",
            "base_url": self.base_url,
            "profiles": self.root.attrib.get('profiles'),
            "min_buffer_time": self.root.attrib.get('minBufferTime'),
            "media_presentation_duration": self.root.attrib.get('mediaPresentationDuration'),
            "adaptation_sets": []
        }

        for adaptation in self.root.findall('.//{*}AdaptationSet', namespaces=self.ns):
            adaptation_info = {
                "id": adaptation.attrib.get('id'),
                "mime_type": adaptation.attrib.get('mimeType'),
                "codecs": adaptation.attrib.get('codecs'),
                "segment_alignment": adaptation.attrib.get('segmentAlignment'),
                "representations": []
            }
            for representation in adaptation.findall('{*}Representation', namespaces=self.ns):
                rep_info = {
                    "id": representation.attrib.get('id'),
                    "bandwidth": representation.attrib.get('bandwidth'),
                    "height": representation.attrib.get('height'),
                    "width": representation.attrib.get('width'),
                    "codecs": representation.attrib.get('codecs'),
                    "base_url": self._get_text_from_child(representation, 'BaseURL'),
                    "segment_template": self._get_segment_template(representation)
                }

                if rep_info["base_url"]:
                    rep_info["base_url"] = resolve_url(self.base_url, rep_info["base_url"])
                resolution = rep_info.get("width"), rep_info.get("height")
                if rep_info.get("height"):
                    quality = f"{rep_info['height']}p"
                else:
                    quality = "unknown"
                rep_info["quality"] = quality
                adaptation_info["representations"].append(rep_info)

            info["adaptation_sets"].append(adaptation_info)
        return info

    def _get_text_from_child(self, parent: ET.Element, child_tag: str) -> Union[str, None]:
        child = parent.find(f'.//{{*}}{child_tag}', namespaces=self.ns)
        if child is not None and child.text:
            return resolve_url(self.base_url, child.text.strip())
        return None

    def _get_segment_template(self, representation: ET.Element) -> Dict[str, Any]:
        segment_template = representation.find('.//{*}SegmentTemplate', namespaces=self.ns)
        if segment_template is not None:
            return {
                "media": resolve_url(self.base_url, segment_template.attrib.get('media', '')),
                "initialization": resolve_url(self.base_url, segment_template.attrib.get('initialization', '')),
                "timescale": segment_template.attrib.get('timescale'),
                "duration": segment_template.attrib.get('duration'),
                "start_number": segment_template.attrib.get('startNumber')
            }
        return {}

# --------------------------------------------------------------------------
# Функции для анализа плейлистов
# --------------------------------------------------------------------------
async def fetch_content(session: aiohttp.ClientSession, url: str) -> Union[str, None]:
    try:
        async with session.get(url, timeout=15) as response:
            if response.status == 200:
                return await response.text()
            else:
                logger.error(f"[fetch_content] Не удалось загрузить {url}: статус {response.status}")
                return None
    except Exception as e:
        logger.error(f"[fetch_content] Ошибка при загрузке {url}: {e}")
        return None

def determine_playlist_type(url: str) -> str:
    if url.lower().endswith('.m3u8'):
        return 'hls'
    elif url.lower().endswith('.mpd'):
        return 'dash'
    return 'unknown'

async def analyze_playlist(url: str) -> Union[Dict[str, Any], None]:
    cleaned_url = clean_url(url)
    playlist_type = determine_playlist_type(cleaned_url)
    if playlist_type == 'unknown':
        logger.warning(f"[analyze_playlist] Неизвестный тип плейлиста: {cleaned_url}")
        return None

    async with aiohttp.ClientSession() as session:
        content = await fetch_content(session, cleaned_url)
        if not content:
            return None

        if playlist_type == 'hls':
            parser = HLSPlaylistParser(content, base_url=cleaned_url)
            return parser.parse()
        elif playlist_type == 'dash':
            parser = DASHPlaylistParser(content, base_url=cleaned_url)
            return parser.parse()
    return None

# --------------------------------------------------------------------------
# Работа с базой данных
# --------------------------------------------------------------------------
import aiosqlite

# NEW: создание таблицы videos
VIDEOS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        site_name TEXT NOT NULL,
        main_link TEXT NOT NULL,
        source_link TEXT NOT NULL,
        last_found TIMESTAMP NOT NULL,
        UNIQUE (main_link, source_link) ON CONFLICT IGNORE
    )
"""

async def init_db(db_name: str = "streams.db", qualities: List[str] = None):
    if qualities is None:
        qualities = ["480p", "720p", "1080p", "1440p", "2160p"]

    async with aiosqlite.connect(db_name) as conn:
        cursor = await conn.cursor()

        # Таблица streams
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS streams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                main_url TEXT NOT NULL UNIQUE,
                nickname TEXT NOT NULL,
                video_url TEXT,
                last_checked TIMESTAMP NOT NULL,
                status TEXT NOT NULL,
                "480p" TEXT,
                "720p" TEXT,
                "1080p" TEXT,
                "1440p" TEXT,
                "2160p" TEXT,
                success BOOLEAN DEFAULT FALSE 
            )
        """)
        logger.info("[DB] Таблица 'streams' создана или уже существует.")

        await cursor.execute("PRAGMA table_info(streams)")
        existing_columns = [info[1] for info in await cursor.fetchall()]

        # Добавляем столбцы для качеств, если нет
        for quality in qualities:
            if quality not in existing_columns:
                try:
                    await cursor.execute(f'ALTER TABLE streams ADD COLUMN "{quality}" TEXT')
                    logger.info(f"[DB] Добавлен столбец для качества: {quality}")
                except aiosqlite.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        logger.debug(f"[DB] Столбец '{quality}' уже существует.")
                    else:
                        logger.error(f"[DB] Ошибка при добавлении столбца '{quality}': {e}")

        # Индекс на nickname (если нужно)
        try:
            await cursor.execute('CREATE INDEX IF NOT EXISTS idx_nickname ON streams (nickname)')
        except aiosqlite.Error as e:
            logger.error(f"[DB] Ошибка создания индекса nickname: {e}")

        # NEW: создаём таблицу videos
        await cursor.execute(VIDEOS_TABLE_SQL)
        logger.info("[DB] Таблица 'videos' создана или уже существует.")

        await conn.commit()

async def update_stream_quality(db_name: str, stream_id: int, quality: str, url: str):
    async with aiosqlite.connect(db_name) as conn:
        cursor = await conn.cursor()
        try:
            await cursor.execute(f"""
                UPDATE streams
                SET "{quality}" = ?
                WHERE id = ?
            """, (url, stream_id))
            if cursor.rowcount == 0:
                logger.warning(f"[DB] Стрим с ID {stream_id} не найден для обновления.")
            else:
                logger.info(f"[DB] Обновлен stream ID {stream_id}: {quality} -> {url}")
            await conn.commit()
        except aiosqlite.Error as e:
            logger.error(f"[DB] Ошибка при обновлении stream ID {stream_id}: {e}")

async def get_stream_by_main_url(db_name: str, main_url: str) -> Union[Dict[str, Any], None]:
    async with aiosqlite.connect(db_name) as conn:
        cursor = await conn.cursor()
        await cursor.execute("""
            SELECT id, video_url, nickname FROM streams
            WHERE main_url = ?
        """, (main_url,))
        row = await cursor.fetchone()
        if row:
            return {"id": row[0], "video_url": row[1], "nickname": row[2]}
        return None

async def save_to_db(original: str, sources: Set[str], db_name: str = "streams.db"):
    """
    Сохранение стрим-ссылок в таблицу streams.
    """
    nickname = extract_nickname(original)
    async with aiosqlite.connect(db_name) as conn:
        cursor = await conn.cursor()
        for source in sources:
            try:
                await cursor.execute("""
                    SELECT id, video_url FROM streams
                    WHERE main_url = ?
                """, (original,))
                row = await cursor.fetchone()
                if row:
                    stream_id, existing_video_url = row
                    if source != existing_video_url:
                        await cursor.execute("""
                            UPDATE streams
                            SET video_url = ?, last_checked = ?, status = 'active', success = TRUE
                            WHERE id = ?
                        """, (source, datetime.now(timezone.utc), stream_id))
                else:
                    await cursor.execute("""
                        INSERT INTO streams (main_url, nickname, video_url, last_checked, status, success)
                        VALUES (?, ?, ?, ?, 'active', TRUE)
                    """, (original, nickname, source, datetime.now(timezone.utc)))
                logger.info(f"[DB] Сохранена ссылка: {source}")
            except aiosqlite.IntegrityError:
                logger.warning(f"[DB] Запись для {original} уже есть (IntegrityError).")
            except aiosqlite.Error as e:
                logger.error(f"[DB] Ошибка вставки данных: {e}")
        await conn.commit()

# NEW: функция для сохранения найденных видео
async def save_videos_to_db(db_name: str, main_link: str, video_links: List[str]):
    """
    Сохраняет обычные видео (не стримы) в таблицу 'videos'.
    Формат: site_name (домен), main_link, source_link, last_found.
    """
    site_name = parse_domain(main_link)
    now = datetime.now(timezone.utc)
    async with aiosqlite.connect(db_name) as conn:
        cursor = await conn.cursor()
        for link in video_links:
            try:
                # INSERT OR IGNORE (см. UNIQUE (main_link, source_link) в таблице)
                await cursor.execute("""
                    INSERT OR IGNORE INTO videos (site_name, main_link, source_link, last_found)
                    VALUES (?, ?, ?, ?)
                """, (site_name, main_link, link, now))
            except aiosqlite.Error as e:
                logger.error(f"[DB] Ошибка при сохранении видео {link}: {e}")
        await conn.commit()

# --------------------------------------------------------------------------
# NetworkWatcher
# --------------------------------------------------------------------------
from playwright.async_api import Page, TimeoutError

class NetworkWatcher:
    def __init__(self):
        self.found_links: Set[str] = set()
        self.is_closed = False
        self.websocket_buffer: Dict[str, str] = {}

    def on_request(self, request):
        if self.is_closed:
            return
        logger.debug(f"[NetworkWatcher] Request: {request.method} {request.url}")

    async def _process_potential_stream(self, url: str, ctype: str = ""):
        if is_stream_url(url, ctype) and not is_placeholder_link(url):
            cleaned = clean_url(url)
            logger.info(f"[NetworkWatcher] Найдена стрим-ссылка: {cleaned}")
            self.found_links.add(cleaned)

    async def on_response(self, response):
        if self.is_closed:
            return
        await self._process_potential_stream(response.url, response.headers.get("content-type", ""))

    async def on_request_finished(self, request):
        if self.is_closed:
            return
        try:
            resp = await request.response()
            if resp:
                await self._process_potential_stream(resp.url, resp.headers.get("content-type", ""))
        except Exception as e:
            logger.error(f"[NetworkWatcher] Ошибка on_request_finished: {e}")

    def on_websocket(self, ws):
        if self.is_closed:
            return
        logger.debug(f"[NetworkWatcher] WebSocket opened: {ws.url}")
        self.websocket_buffer[ws.url] = ""
        ws.on("framereceived", lambda frame: asyncio.create_task(self.on_websocket_frame(ws, frame)))
        ws.on("framesent", lambda frame: asyncio.create_task(self.on_websocket_frame(ws, frame)))
        ws.on("close", lambda frame: asyncio.create_task(self.on_websocket_close(ws)))

    async def on_websocket_frame(self, ws, frame):
        if self.is_closed:
            return
        try:
            if isinstance(frame, dict):
                opcode = frame.get("opcode")
                payload = frame.get("payload", "")
                if opcode == 1:  # Текстовый фрейм
                    if isinstance(payload, bytes):
                        try:
                            text_data = payload.decode("utf-8", errors="ignore")
                        except UnicodeDecodeError:
                            logger.error("[NetworkWatcher/WebSocket] Ошибка декодирования.")
                            return
                    elif isinstance(payload, str):
                        text_data = payload
                    else:
                        logger.warning(f"Неизвестный тип payload: {type(payload)}")
                        return
                    logger.debug(f"[NetworkWatcher/WebSocket] Получено сообщение: {text_data}")
                    self.websocket_buffer[ws.url] += text_data
                elif opcode == 8:
                    await self.on_websocket_close(ws)
            elif isinstance(frame, str):
                logger.debug(f"Получено строковое сообщение: {frame}")
                self.websocket_buffer[ws.url] += frame
            else:
                logger.warning(f"Неожиданный тип фрейма: {type(frame)}")
        except (ConnectionResetError) as e:
            logger.error(f"[NetworkWatcher/WebSocket] Ошибка соединения: {e}")

    async def on_websocket_close(self, ws):
        if self.is_closed:
            return
        if ws.url in self.websocket_buffer:
            data = self.websocket_buffer[ws.url]
            logger.debug(f"[NetworkWatcher/WebSocket] Закрытие, анализ буфера: {ws.url}")
            found = URL_PATTERN.findall(data)
            for url_ in found:
                await self._process_potential_stream(url_)
            del self.websocket_buffer[ws.url]

    def close(self):
        self.is_closed = True
        self.websocket_buffer.clear()

# --------------------------------------------------------------------------
# PageParser
# --------------------------------------------------------------------------
from bs4 import BeautifulSoup

class PageParser:
    """
    Собирает title, ссылки в <video>, <source>, а также ищет <script> + JSON.
    """
    def __init__(self, page: Page):
        self.page = page
        self.results: Dict[str, Any] = {
            "page_title": None,
            "videos": [],   # <video src="...">
            "sources": [],  # <source src="...">
            "script_urls": []
        }

    async def parse(self) -> Dict[str, Any]:
        await self._parse_frame(self.page.main_frame)
        other_frames = [fr for fr in self.page.frames if fr != self.page.main_frame]
        if other_frames:
            await asyncio.gather(*[self._parse_frame(fr) for fr in other_frames])
        return self.results

    async def _parse_frame(self, frame):
        try:
            html = await frame.content()
        except Exception as e:
            logger.error(f"[PageParser] Ошибка получения HTML: {e}")
            return

        soup = BeautifulSoup(decode_js_escapes(html), "html.parser")

        if frame == self.page.main_frame:
            title_tag = soup.find("title")
            if title_tag:
                self.results["page_title"] = title_tag.get_text(strip=True)
                logger.info(f"[PageParser] Заголовок: {self.results['page_title']}")

        for vid_tag in soup.find_all("video"):
            s = vid_tag.get("src", "")
            if s:
                cleaned = clean_url(s)
                if not is_placeholder_link(cleaned):
                    self.results["videos"].append(cleaned)
                    logger.info(f"[PageParser] Найдена видео-ссылка: {cleaned}")

        for src_tag in soup.find_all("source"):
            s = src_tag.get("src", "")
            if s:
                cleaned = clean_url(s)
                if not is_placeholder_link(cleaned):
                    self.results["sources"].append(cleaned)
                    logger.info(f"[PageParser] Найден source: {cleaned}")

        # Ищем <script>, ищем JSON/URL
        for sc in soup.find_all("script"):
            stxt = decode_js_escapes(sc.get_text() or "").strip()
            # При наличии ключевых слов пробуем распознать JSON
            if any(key in stxt.lower() for key in ["manifest", "stream", "url", "hls", "mpd", ".m3u8", ".mpd"]):
                if (stxt.startswith('{') and stxt.endswith('}')) or \
                   (stxt.startswith('[') and stxt.endswith(']')):
                    try:
                        data = json.loads(stxt)
                        found_urls = deep_search_for_urls(data)
                        for found_url in found_urls:
                            cleaned = clean_url(found_url)
                            if is_stream_url(cleaned, "") and not is_placeholder_link(cleaned):
                                self.results["script_urls"].append(cleaned)
                                logger.info(f"[PageParser] Стрим из JSON: {cleaned}")
                    except json.JSONDecodeError:
                        logger.debug("[PageParser] Не валидный JSON.")
            found = URL_PATTERN.findall(stxt)
            for url_ in found:
                cleaned = clean_url(url_)
                if is_stream_url(cleaned, "") and not is_placeholder_link(cleaned):
                    self.results["script_urls"].append(cleaned)
                    logger.info(f"[PageParser] Стрим-ссылка в script: {cleaned}")

# --------------------------------------------------------------------------
# Клик по кнопкам "Play"
# --------------------------------------------------------------------------
async def click_play_buttons(page: Page):
    play_selectors = [
        "button.play",
        ".play-button",
        "button#play",
        "button[aria-label='Play']",
        ".btn-play",
        "button.play-button",
        "div.play-button",
        "button[data-testid='play-button']",
        "div[data-testid='play-button']",
        "video"
    ]
    clicked = False
    for sel in play_selectors:
        try:
            await page.click(sel, timeout=500)
            logger.debug(f"[click_play_buttons] Clicked {sel}")
            clicked = True
            break
        except Exception:
            logger.debug(f"[click_play_buttons] Не удалось кликнуть: {sel}")

    if not clicked:
        for fr in page.frames:
            if fr != page.main_frame:
                try:
                    await fr.click("video", timeout=500)
                    logger.debug("[click_play_buttons] Clicked <video> in iframe")
                    break
                except Exception:
                    logger.debug("[click_play_buttons] Не кликнулось во фрейме")

    try:
        await page.evaluate("() => { document.querySelectorAll('video').forEach(v => v.play().catch(()=>{})) }")
        logger.debug("[click_play_buttons] video.play() вызвано")
    except Exception as e:
        logger.error(f"[click_play_buttons] Ошибка video.play(): {e}")

    try:
        await page.mouse.move(100, 100)
        await page.mouse.wheel(0, 500)
        logger.debug("[click_play_buttons] Эмуляция скролла")
    except Exception as e:
        logger.error(f"[click_play_buttons] Ошибка эмуляции действий: {e}")

# --------------------------------------------------------------------------
# Функция для вывода результатов (консоль)
# --------------------------------------------------------------------------
def display_results(site: str, found: Set[str], results: Dict[str, Any]):
    print(f"\n=== РЕЗУЛЬТАТ ДЛЯ {site} ===")
    print(f"Заголовок страницы: {results.get('page_title')}")
    if found:
        print(f"Найдены потоки ({len(found)}):")
        for link in found:
            print("  ", link)
    else:
        print("Потоки не найдены.")
    print("\n--- Детали ---")
    if results.get("videos"):
        print("Видео:", results["videos"])
    if results.get("sources"):
        print("Source:", results["sources"])
    if results.get("script_urls"):
        print("Script:", results["script_urls"])
    print("-" * 30, "\n")

# --------------------------------------------------------------------------
# Проверка доступности URL (HEAD)
# --------------------------------------------------------------------------
async def is_url_valid(url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=10) as resp:
                return resp.status == 200
    except Exception:
        return False

# --------------------------------------------------------------------------
# SiteCrawler
# --------------------------------------------------------------------------
from playwright.async_api import async_playwright

class SiteCrawler:
    def __init__(
            self,
            start_url: str,
            max_depth: int = 3,
            exclude_patterns: List[str] = None,
            db_name: str = "streams.db",
            popup_selectors: List[Dict[str, Any]] = None,
            preview_image_pattern: str = r"https://thumb\.live\.mmcdn\.com/.*\.png",
            content_class: str = "content",
            concurrency: int = 5,
            global_concurrency: int = 10,
            inactive_check_interval: int = 3600,
            max_retries: int = 3,
            visited_urls_limit: int = 1000,
            loop_prevention_threshold: int = 5,
            db_updated_callback=None
    ):
        self.start_url = clean_url(start_url)
        self.max_depth = max_depth
        self.db_name = db_name
        self.visited_urls: Set[str] = set()
        self.to_visit: asyncio.Queue = asyncio.Queue()
        self.to_visit.put_nowait((self.start_url, 0))
        self.preview_image_pattern = re.compile(preview_image_pattern, re.IGNORECASE)
        self.content_class = content_class
        self.concurrency = concurrency
        self.global_concurrency = global_concurrency
        self.inactive_check_interval = inactive_check_interval
        self.max_retries = max_retries
        self.visited_urls_limit = visited_urls_limit
        self.playwright = None
        self.browser = None
        self.db_updated_callback = db_updated_callback

        self.exclude_patterns = exclude_patterns or [
            "settings", "swag", "terms", "privacy", "conditions", "support",
            "dmca", "remove-content", "feedback", "security-center",
            "law-enforcement", "report", "nonconsensual", "abusive-content",
            "billing", "disable-account", "apps", "contest", "affiliates",
            "jobs", "sitemap", "language"
        ]
        self.popup_selectors = popup_selectors or [
            {"selector": "button.close, button.close-button, .modal-close, .close-modal", "action": "click"},
            {"selector": "button.accept, button.agree, button#accept, button#agree", "action": "click"},
            {"selector": "button.confirm, button#confirm", "action": "click"},
            {"selector": "button#age-confirm, button#yes", "action": "click"},
        ]
        self.loop_prevention_threshold = loop_prevention_threshold
        self.url_visit_counts: Dict[str, int] = {}

        self.semaphore = asyncio.Semaphore(concurrency)
        # Инициализация таблиц
        asyncio.create_task(init_db(self.db_name))

    def should_exclude(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.lower()
        for pattern in self.exclude_patterns:
            if pattern.lower() in path:
                return True
        return False

    def is_same_domain(self, base_url: str, other_url: str) -> bool:
        base_domain = urllib.parse.urlparse(base_url).netloc
        other_domain = urllib.parse.urlparse(other_url).netloc
        return base_domain == other_domain

    async def handle_popups(self, page: Page):
        for popup in self.popup_selectors:
            selector = popup["selector"]
            action = popup.get("action", "click")
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    if action == "click":
                        await element.click(timeout=3000)
                    elif action == "accept":
                        await element.click(timeout=3000)
                if elements:
                    logger.info(f"[Popup Handler] Закрыл(а) {selector}, action={action}")
            except Exception as e:
                logger.debug(f"[Popup Handler] Не удалось обработать {selector}: {e}")

    async def click_preview_images(self, page: Page):
        clicked_images: Set[str] = set()
        try:
            preview_imgs = await page.query_selector_all("img[src]")
            for img in preview_imgs:
                src = await img.get_attribute("src")
                if src and self.preview_image_pattern.match(src) and src not in clicked_images:
                    try:
                        await img.click(timeout=3000)
                        clicked_images.add(src)
                        logger.info(f"[Preview Clicker] Клик по превью: {src}")
                        await page.wait_for_load_state("networkidle", timeout=5000)
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"[Preview Clicker] Ошибка клика: {e}")
        except Exception as e:
            logger.error(f"[Preview Clicker] Ошибка поиска превью: {e}")

    async def extract_internal_links(self, page: Page) -> Set[str]:
        internal_links = set()
        try:
            content_divs = await page.query_selector_all(f"div.{self.content_class}")
            for content_div in content_divs:
                link_elems = await content_div.query_selector_all("a[href]")
                for link in link_elems:
                    href = await link.get_attribute("href")
                    if href:
                        abs_url = urllib.parse.urljoin(self.start_url, href)
                        abs_url = clean_url(abs_url)
                        if self.is_same_domain(self.start_url, abs_url):
                            internal_links.add(abs_url)
            logger.info(f"[Crawler] Найдено {len(internal_links)} внутренних ссылок в '{self.content_class}'")
        except Exception as e:
            logger.error(f"[Crawler] Ошибка извлечения внутренних ссылок: {e}")
        return internal_links

    async def update_database(self, main_url: str, video_urls: Set[str]):
        """Сохраняем стрим-ссылки в streams."""
        try:
            await save_to_db(main_url, video_urls, db_name=self.db_name)
            if self.db_updated_callback:
                self.db_updated_callback()
        except Exception as e:
            logger.error(f"[Crawler] Ошибка update_database: {e}")

    # NEW:
    async def save_videos(self, main_url: str, normal_videos: List[str]):
        """
        Сохраняет обычные «не-стримовые» ссылки в таблицу videos.
        """
        if normal_videos:
            try:
                await save_videos_to_db(self.db_name, main_url, normal_videos)
            except Exception as e:
                logger.error(f"[Crawler] Ошибка save_videos: {e}")

    async def process_streams(self, video_urls: Set[str], main_url: str):
        async with aiohttp.ClientSession() as session:
            tasks = [self.process_stream(url, main_url, session) for url in video_urls]
            await asyncio.gather(*tasks)

    async def process_stream(self, stream_url: str, main_url: str, session: aiohttp.ClientSession):
        try:
            info = await analyze_playlist(stream_url)
            if not info:
                logger.warning(f"[Process Stream] Не удалось проанализировать {stream_url}")
                return
            extracted_qualities = {}
            if info["type"] == "HLS":
                for pl in info.get("playlists", []):
                    q = pl.get("quality")
                    uri = pl.get("uri")
                    if q and uri:
                        extracted_qualities[q] = uri
            elif info["type"] == "DASH":
                for adap_set in info.get("adaptation_sets", []):
                    for rep in adap_set.get("representations", []):
                        q = rep.get("quality")
                        media_url = rep.get("segment_template", {}).get("media")
                        if q and media_url:
                            extracted_qualities[q] = media_url

            record = await get_stream_by_main_url(self.db_name, main_url)
            if not record:
                logger.warning(f"[Process Stream] Нет записи для {main_url}")
                return
            stream_id = record["id"]

            async with aiosqlite.connect(self.db_name) as conn:
                cursor = await conn.cursor()
                for qual, url_ in extracted_qualities.items():
                    if qual in ["480p", "720p", "1080p", "1440p", "2160p"]:
                        await cursor.execute(f'UPDATE streams SET "{qual}" = ? WHERE id = ?', (url_, stream_id))
                await conn.commit()
                if self.db_updated_callback:
                    self.db_updated_callback()
        except Exception as e:
            logger.error(f"[Process Stream] Ошибка {stream_url}: {e}")

    async def crawl_url(self, page: Page, current_url: str, depth: int):
        try:
            logger.info(f"[Crawler] URL={current_url}, depth={depth}")
            self.visited_urls.add(current_url)

            for attempt in range(self.max_retries):
                try:
                    await page.goto(current_url, timeout=30000)
                    break
                except TimeoutError:
                    logger.warning(f"[Crawler] Таймаут: {current_url} (попытка {attempt+1})")
                    if attempt == self.max_retries - 1:
                        raise

            await self.handle_popups(page)
            internal_links = await self.extract_internal_links(page)
            for link in internal_links:
                if link not in self.visited_urls:
                    parsed = urllib.parse.urlparse(link)
                    base_path = parsed.path.split('/')[1] if '/' in parsed.path else ''
                    if base_path:
                        self.url_visit_counts[base_path] = self.url_visit_counts.get(base_path, 0) + 1
                        if self.url_visit_counts[base_path] <= self.loop_prevention_threshold:
                            await self.to_visit.put((link, depth + 1))

            await self.click_preview_images(page)
            await self.handle_popups(page)

            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
            except TimeoutError:
                logger.warning("[Crawler] networkidle не дождался")

            parser = PageParser(page)
            nw = NetworkWatcher()
            page.on("response", lambda r: asyncio.create_task(nw.on_response(r)))
            page.on("requestfinished", lambda r: asyncio.create_task(nw.on_request_finished(r)))
            page.on("websocket", nw.on_websocket)
            page.on("request", nw.on_request)

            max_attempts = 2
            found_streams = set()
            for attempt in range(1, max_attempts + 1):
                await click_play_buttons(page)
                await asyncio.sleep(2)
                current_results = await parser.parse()

                # Собираем стрим-ссылки (с учётом NetworkWatcher)
                video_candidates = set(current_results["videos"] + current_results["sources"] + current_results["script_urls"])
                video_candidates.update(nw.found_links)

                valid_streams = set()
                for c in video_candidates:
                    c_clean = clean_url(c)
                    if is_stream_url(c_clean, "") and not is_placeholder_link(c_clean):
                        if await is_url_valid(c_clean):
                            valid_streams.add(c_clean)

                if valid_streams:
                    found_streams = valid_streams
                    logger.info("[Crawler] Стримы найдены, завершаем попытки")
                    break
                else:
                    logger.info("[Crawler] Стримы не найдены, следующая попытка...")

            # CHANGED: теперь сохраняем и «обычные» видео
            # Выделим среди videos/sources то, что не является стримом
            normal_videos = []
            for link_ in (current_results["videos"] + current_results["sources"]):
                cleaned_ = clean_url(link_)
                if cleaned_ and not is_placeholder_link(cleaned_) and not is_stream_url(cleaned_):
                    normal_videos.append(cleaned_)

            # Сохраняем стримы и обычные видео
            await self.update_database(current_url, found_streams)
            await self.save_videos(current_url, normal_videos)

            parse_results = {
                "page_title": current_results["page_title"],
                "videos": current_results["videos"],
                "sources": current_results["sources"],
                "script_urls": current_results["script_urls"]
            }
            display_results(current_url, found_streams, parse_results)

            await self.process_streams(found_streams, current_url)

        except Exception as e:
            logger.error(f"[Crawler] Ошибка: {current_url}: {e}")
        finally:
            await page.close()

    async def crawl(self):
        async with async_playwright() as p:
            self.playwright = p
            self.browser = await p.chromium.launch(headless=True)
            context = await self.browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/111.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
            )
            workers = [asyncio.create_task(self.worker(context)) for _ in range(self.concurrency)]
            await asyncio.gather(*workers)
            await context.close()
            await self.browser.close()

    async def worker(self, context):
        while True:
            if self.to_visit.empty():
                has_inactive = await self.check_inactive_streams()
                if not has_inactive:
                    break
            try:
                current_url, depth = self.to_visit.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.1)
                continue

            if (current_url in self.visited_urls) or self.should_exclude(current_url) or (depth > self.max_depth):
                self.to_visit.task_done()
                continue

            if len(self.visited_urls) >= self.visited_urls_limit:
                logger.warning(f"[Crawler] Лимит посещённых URL достигнут: {self.visited_urls_limit}")
                break

            await self.semaphore.acquire()
            try:
                page = await context.new_page()
                await self.crawl_url(page, current_url, depth)
            finally:
                self.semaphore.release()
                self.to_visit.task_done()

    async def check_inactive_streams(self):
        """
        Демонстрация: можно искать неактивные стримы (status='inactive') и переобходить
        """
        try:
            async with aiosqlite.connect(self.db_name) as conn:
                cursor = await conn.cursor()
                await cursor.execute("""
                    SELECT main_url
                    FROM streams
                    WHERE status = 'inactive' AND success = TRUE AND last_checked < ?
                """, (datetime.now(timezone.utc) - timedelta(seconds=self.inactive_check_interval),))
                rows = await cursor.fetchall()
                if rows:
                    for row in rows:
                        await self.to_visit.put((row[0], 0))
                    logger.info(f"[Crawler] Добавлено {len(rows)} 'inactive' стримов.")
                    return True
        except aiosqlite.Error as e:
            logger.error(f"[Crawler] Ошибка inactive_streams: {e}")
        return False

# --------------------------------------------------------------------------
# Консольный запуск (опционально)
# --------------------------------------------------------------------------
async def console_main():
    """ Запуск в консольном режиме. """
    await init_db()
    urls = []

    if len(sys.argv) > 2 and sys.argv[1] == "--file":
        file_path = sys.argv[2]
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            urls.extend(lines)

    if len(sys.argv) > 1 and not urls:
        if sys.argv[1] != "--file":
            urls = sys.argv[1:]
    else:
        if not urls:
            print("Введите ссылки (через пробел):")
            inp = input()
            urls = inp.strip().split()

    if not urls:
        print("Нет ссылок, завершаем")
        return

    if len(urls) == 1:
        c = SiteCrawler(urls[0], max_depth=3)
        await c.crawl()
    else:
        crawlers = [SiteCrawler(u, max_depth=3) for u in urls]
        await asyncio.gather(*(cr.crawl() for cr in crawlers))

# --------------------------------------------------------------------------
# Перенаправление вывода в ScrolledText
# --------------------------------------------------------------------------
class TextRedirector:
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, s):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, s)
        self.widget.configure(state='disabled')
        self.widget.see(tk.END)

    def flush(self):
        pass

# --------------------------------------------------------------------------
# GUI: создаём дополнительную вкладку "Videos DB" для новой таблицы
# --------------------------------------------------------------------------
class CrawlerGUI:
    def __init__(self, master):
        self.vlc_path = tk.StringVar(value=self.find_vlc_path())

        self.master = master
        master.title("Стрим-Скрапер")

        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat")
        style.configure("Treeview", rowheight=25)

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill="both", expand=True)

        self.crawl_frame = ttk.Frame(self.notebook)
        self.db_frame = ttk.Frame(self.notebook)
        # NEW: вкладка "Videos"
        self.videos_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.crawl_frame, text="Scraping")
        self.notebook.add(self.db_frame, text="Streams DB")
        self.notebook.add(self.videos_frame, text="Videos DB")  # NEW

        self.crawler_thread = None
        self.crawlers: List[SiteCrawler] = []

        self.create_crawl_widgets()
        self.create_db_widgets()
        self.create_videos_widgets()  # NEW

        asyncio.run(init_db("streams.db"))  # Инициализация БД

        self.load_db_data()
        self.load_videos_data()  # NEW

    def create_crawl_widgets(self):
        self.url_label = ttk.Label(self.crawl_frame, text="URL (через пробел):")
        self.url_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.url_entry = ttk.Entry(self.crawl_frame, width=50)
        self.url_entry.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
        self.url_entry.insert(0, "https://www.google.com")

        self.load_file_button = ttk.Button(self.crawl_frame, text="Загрузить из файла", command=self.load_urls_from_file)
        self.load_file_button.grid(row=0, column=4, padx=5, pady=5, sticky="ew")

        self.depth_label = ttk.Label(self.crawl_frame, text="Depth (0 - 3):")
        self.depth_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.depth_var = tk.IntVar(value=3)
        self.depth_entry = ttk.Entry(self.crawl_frame, textvariable=self.depth_var, width=5)
        self.depth_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        self.concurrency_label = ttk.Label(self.crawl_frame, text="Concurrency (1 - 10):")
        self.concurrency_label.grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.concurrency_var = tk.IntVar(value=5)
        self.concurrency_entry = ttk.Entry(self.crawl_frame, textvariable=self.concurrency_var, width=5)
        self.concurrency_entry.grid(row=1, column=3, sticky="w", padx=5, pady=5)

        self.vlc_path_label = ttk.Label(self.crawl_frame, text="Path to VLC:")
        self.vlc_path_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.vlc_path_entry = ttk.Entry(self.crawl_frame, textvariable=self.vlc_path, width=40)
        self.vlc_path_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        self.vlc_path_button = ttk.Button(self.crawl_frame, text="Browse", command=self.browse_vlc_path)
        self.vlc_path_button.grid(row=2, column=3, padx=5, pady=5, sticky="ew")

        self.start_button = ttk.Button(self.crawl_frame, text="Start", command=self.start_scraping)
        self.start_button.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        self.stop_button = ttk.Button(self.crawl_frame, text="Stop", command=self.stop_scraping, state="disabled")
        self.stop_button.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        self.refresh_db_button = ttk.Button(self.crawl_frame, text="Refresh Streams DB", command=self.refresh_db)
        self.refresh_db_button.grid(row=3, column=2, padx=5, pady=5, sticky="ew")

        self.clear_log_button = ttk.Button(self.crawl_frame, text="Clear Logs", command=self.clear_log)
        self.clear_log_button.grid(row=3, column=3, padx=5, pady=5, sticky="ew")

        self.save_log_button = ttk.Button(self.crawl_frame, text="Save Logs", command=self.save_log)
        self.save_log_button.grid(row=3, column=4, padx=5, pady=5, sticky="ew")

        self.log_label = ttk.Label(self.crawl_frame, text="Log:")
        self.log_label.grid(row=4, column=0, sticky="w", padx=5, pady=5)

        self.log_area = scrolledtext.ScrolledText(self.crawl_frame, wrap=tk.WORD, height=20)
        self.log_area.grid(row=5, column=0, columnspan=5, padx=5, pady=5, sticky="nsew")
        self.log_area.configure(state='disabled')

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.crawl_frame, orient="horizontal", length=300, mode="indeterminate", variable=self.progress_var)
        self.progress_bar.grid(row=6, column=0, columnspan=5, padx=5, pady=10, sticky="ew")

        self.crawl_frame.columnconfigure(1, weight=1)
        self.crawl_frame.rowconfigure(5, weight=1)

        sys.stdout = TextRedirector(self.log_area, "stdout")
        sys.stderr = TextRedirector(self.log_area, "stderr")

    def load_urls_from_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите файл со списком URL",
            filetypes=[("Текстовые файлы", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                if lines:
                    current_text = self.url_entry.get().strip()
                    if current_text:
                        combined_urls = current_text.split() + lines
                    else:
                        combined_urls = lines
                    new_text = ' '.join(combined_urls)
                    self.url_entry.delete(0, tk.END)
                    self.url_entry.insert(0, new_text)
                    messagebox.showinfo("Загрузка завершена", f"Загружено {len(lines)} URL из файла.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось прочитать файл: {e}")

    def browse_vlc_path(self):
        file_path = filedialog.askopenfilename(
            initialdir=".",
            title="Select vlc.exe",
            filetypes=(("Executable files", "*.exe"), ("All files", "*.*"))
        )
        if file_path:
            self.vlc_path.set(file_path)

    def find_vlc_path(self):
        """Пытается найти VLC на Windows."""
        vlc_path = None
        if sys.platform == "win32":
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\VideoLAN\VLC") as key:
                    install_dir = winreg.QueryValueEx(key, "InstallDir")[0]
                    vlc_path = os.path.join(install_dir, "vlc.exe")
            except FileNotFoundError:
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Wow6432Node\VideoLAN\VLC") as key:
                        install_dir = winreg.QueryValueEx(key, "InstallDir")[0]
                        vlc_path = os.path.join(install_dir, "vlc.exe")
                except FileNotFoundError:
                    pass
            if not vlc_path:
                program_files_paths = [
                    os.environ.get("ProgramFiles"),
                    os.environ.get("ProgramFiles(x86)"),
                    os.environ.get("ProgramW6432")
                ]
                for pf_path in program_files_paths:
                    if pf_path:
                        default_path = os.path.join(pf_path, "VideoLAN", "VLC", "vlc.exe")
                        if os.path.exists(default_path):
                            vlc_path = default_path
                            break
        return vlc_path

    def create_db_widgets(self):
        self.search_label = ttk.Label(self.db_frame, text="Search by nickname:")
        self.search_label.pack(pady=5, anchor='w', padx=5)

        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(self.db_frame, textvariable=self.search_var)
        self.search_entry.pack(pady=5, fill='x', padx=5)
        self.search_entry.bind("<KeyRelease>", self.search_db)

        self.db_tree = ttk.Treeview(
            self.db_frame,
            columns=("id", "main_url", "nickname", "video_url", "480p", "720p", "1080p", "1440p", "2160p", "last_checked", "status"),
            show="headings"
        )
        self.db_tree.pack(fill="both", expand=True)

        columns = ["id", "main_url", "nickname", "video_url", "480p", "720p", "1080p", "1440p", "2160p", "last_checked", "status"]
        for col in columns:
            self.db_tree.heading(col, text=col.capitalize(), command=lambda c=col: self.sort_streams_column(c, False))

        self.db_tree.column("id", width=30, anchor="center")
        self.db_tree.column("main_url", anchor="w", width=200)
        self.db_tree.column("nickname", anchor="w", width=150)
        self.db_tree.column("video_url", anchor="w", width=200)
        self.db_tree.column("480p", anchor="w", width=80)
        self.db_tree.column("720p", anchor="w", width=80)
        self.db_tree.column("1080p", anchor="w", width=80)
        self.db_tree.column("1440p", anchor="w", width=80)
        self.db_tree.column("2160p", anchor="w", width=80)
        self.db_tree.column("last_checked", width=120, anchor="center")
        self.db_tree.column("status", width=80, anchor="center")

        self.db_tree.bind("<Button-3>", self.show_streams_context_menu)

        # Кнопка экспорта streams, если нужно
        # (Пользователь просил экспорт именно "videos", поэтому ниже сделаем для видео)

    # NEW: вкладка Video DB
    def create_videos_widgets(self):
        # Таблица videos
        self.video_tree = ttk.Treeview(
            self.videos_frame,
            columns=("id", "site_name", "main_link", "source_link", "last_found"),
            show="headings"
        )
        self.video_tree.pack(fill="both", expand=True)

        for col in ("id", "site_name", "main_link", "source_link", "last_found"):
            self.video_tree.heading(col, text=col.capitalize(), command=lambda c=col: self.sort_videos_column(c, False))
        self.video_tree.column("id", width=40, anchor="center")
        self.video_tree.column("site_name", width=120, anchor="w")
        self.video_tree.column("main_link", width=200, anchor="w")
        self.video_tree.column("source_link", width=200, anchor="w")
        self.video_tree.column("last_found", width=120, anchor="center")

        self.video_tree.bind("<Button-3>", self.show_videos_context_menu)

        # Кнопка экспорта
        self.export_videos_button = ttk.Button(self.videos_frame, text="Export Videos to Text", command=self.export_videos)
        self.export_videos_button.pack(pady=5, anchor='e')

    def show_streams_context_menu(self, event):
        region = self.db_tree.identify_region(event.x, event.y)
        if region == "cell":
            item = self.db_tree.identify_row(event.y)
            column = self.db_tree.identify_column(event.x)
            col_id = int(column.replace("#", "")) - 1
            if item and column:
                self.db_tree.selection_set(item)
                menu = tk.Menu(self.master, tearoff=0)
                cell_value = self.db_tree.item(item, "values")[col_id]
                if cell_value and isinstance(cell_value, str) and cell_value.startswith("http"):
                    menu.add_command(label="Open in VLC", command=lambda: self.open_in_vlc(cell_value))
                    menu.add_command(label="Open in Browser", command=lambda: self.open_in_browser(cell_value))
                menu.add_command(label="Copy", command=lambda i=item, c=col_id: self.copy_streams_cell(i, c))
                menu.post(event.x_root, event.y_root)
                menu.grab_release()

    def show_videos_context_menu(self, event):
        region = self.video_tree.identify_region(event.x, event.y)
        if region == "cell":
            item = self.video_tree.identify_row(event.y)
            column = self.video_tree.identify_column(event.x)
            col_id = int(column.replace("#", "")) - 1
            if item and column:
                self.video_tree.selection_set(item)
                menu = tk.Menu(self.master, tearoff=0)
                cell_value = self.video_tree.item(item, "values")[col_id]
                # Позволяем открыть, если это ссылка
                if col_id == 2 or col_id == 3:  # main_link или source_link
                    if isinstance(cell_value, str) and cell_value.startswith("http"):
                        menu.add_command(label="Open in Browser", command=lambda: self.open_in_browser(cell_value))
                        menu.add_command(label="Open in VLC", command=lambda: self.open_in_vlc(cell_value))
                menu.add_command(label="Copy", command=lambda i=item, c=col_id: self.copy_videos_cell(i, c))
                menu.post(event.x_root, event.y_root)
                menu.grab_release()

    def copy_streams_cell(self, item, col_id):
        val = self.db_tree.item(item, "values")[col_id]
        self.master.clipboard_clear()
        self.master.clipboard_append(val)
        messagebox.showinfo("Скопировано", "Содержимое ячейки скопировано.")

    def copy_videos_cell(self, item, col_id):
        val = self.video_tree.item(item, "values")[col_id]
        self.master.clipboard_clear()
        self.master.clipboard_append(val)
        messagebox.showinfo("Скопировано", "Содержимое ячейки скопировано.")

    def open_in_vlc(self, url):
        try:
            vlc_path = self.vlc_path.get()
            if vlc_path and os.path.exists(vlc_path):
                subprocess.Popen([vlc_path, url])
            else:
                webbrowser.open(url)
                messagebox.showwarning("VLC not found", "VLC не найден. Открываю в браузере.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open in VLC: {e}")

    def open_in_browser(self, url):
        try:
            webbrowser.open(url)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open in browser: {e}")

    def refresh_db(self):
        if messagebox.askyesno("Confirmation", "Обновить все ссылки из Streams DB?"):
            self.progress_bar.start()
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.refresh_db_button.configure(state="disabled")
            th = threading.Thread(target=self.run_db_refresh)
            th.daemon = True
            th.start()

    def run_db_refresh(self):
        async def update_db_async():
            try:
                async with aiosqlite.connect("streams.db") as conn:
                    cursor = await conn.cursor()
                    await cursor.execute("SELECT main_url FROM streams")
                    rows = await cursor.fetchall()
                urls = [r[0] for r in rows]
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    context = await browser.new_context(
                        user_agent=(
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/111.0.0.0 Safari/537.36"
                        ),
                        viewport={"width": 1280, "height": 800},
                    )
                    async def check_url(url):
                        try:
                            page = await context.new_page()
                            nw = NetworkWatcher()
                            page.on("response", lambda r: asyncio.create_task(nw.on_response(r)))
                            page.on("requestfinished", lambda r: asyncio.create_task(nw.on_request_finished(r)))
                            page.on("websocket", nw.on_websocket)
                            page.on("request", nw.on_request)

                            crawler = SiteCrawler(url, max_depth=0, db_name="streams.db", db_updated_callback=self.update_gui_db_table)
                            await crawler.crawl_url(page, url, 0)
                            await page.close()
                        except Exception as e:
                            logger.error(f"Ошибка при обновлении {url}: {e}")
                    tasks = [check_url(u) for u in urls]
                    await asyncio.gather(*tasks)
                    await context.close()
                    await browser.close()
            except Exception as e:
                logger.error(f"Ошибка refresh DB: {e}")
            finally:
                self.master.after(0, self.finish_db_refresh)

        asyncio.run(update_db_async())

    def finish_db_refresh(self):
        self.progress_bar.stop()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.refresh_db_button.configure(state="normal")
        self.update_gui_db_table()
        messagebox.showinfo("Готово", "Streams DB обновлена!")

    def update_gui_db_table(self):
        self.master.after(0, self._update_streams_table_task)

    def _update_streams_table_task(self):
        self.db_tree.delete(*self.db_tree.get_children())
        self.load_db_data()

        # NEW: обновим и видео-таблицу
        self.video_tree.delete(*self.video_tree.get_children())
        self.load_videos_data()

    def sort_streams_column(self, col, reverse):
        data = [(self.db_tree.set(child, col), child) for child in self.db_tree.get_children('')]
        try:
            if col == "id":
                data.sort(key=lambda x: int(x[0]), reverse=reverse)
            elif col == "last_checked":
                data.sort(key=lambda x: datetime.fromisoformat(x[0]), reverse=reverse)
            else:
                data.sort(key=lambda x: str(x[0]).lower(), reverse=reverse)
        except Exception as e:
            logger.error(f"Ошибка сортировки: {e}")
            data.sort(reverse=reverse)
        for index, (val, item) in enumerate(data):
            self.db_tree.move(item, '', index)
        self.db_tree.heading(col, command=lambda c=col: self.sort_streams_column(c, not reverse))

    # NEW: сортировка для videos
    def sort_videos_column(self, col, reverse):
        data = [(self.video_tree.set(child, col), child) for child in self.video_tree.get_children('')]
        try:
            if col == "id":
                data.sort(key=lambda x: int(x[0]), reverse=reverse)
            elif col == "last_found":
                data.sort(key=lambda x: datetime.fromisoformat(x[0]), reverse=reverse)
            else:
                data.sort(key=lambda x: str(x[0]).lower(), reverse=reverse)
        except Exception as e:
            logger.error(f"Ошибка сортировки videos: {e}")
            data.sort(reverse=reverse)
        for index, (val, item) in enumerate(data):
            self.video_tree.move(item, '', index)
        self.video_tree.heading(col, command=lambda c=col: self.sort_videos_column(c, not reverse))

    async def _load_streams(self):
        async with aiosqlite.connect("streams.db") as conn:
            cursor = await conn.cursor()
            try:
                await cursor.execute("""
                    SELECT id, main_url, nickname, video_url, "480p", "720p", "1080p", "1440p", "2160p", last_checked, status
                    FROM streams
                """)
                rows = await cursor.fetchall()
                for row in rows:
                    self.db_tree.insert("", "end", values=row)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить streams: {e}")

    # NEW: загрузка из таблицы videos
    async def _load_videos(self):
        async with aiosqlite.connect("streams.db") as conn:
            cursor = await conn.cursor()
            try:
                await cursor.execute("""
                    SELECT id, site_name, main_link, source_link, last_found
                    FROM videos
                """)
                rows = await cursor.fetchall()
                for row in rows:
                    self.video_tree.insert("", "end", values=row)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить videos: {e}")

    def load_db_data(self):
        asyncio.run(self._load_streams())

    # NEW:
    def load_videos_data(self):
        asyncio.run(self._load_videos())

    def search_db(self, event):
        search_text = self.search_var.get().strip().lower()
        async def async_search():
            async with aiosqlite.connect("streams.db") as conn:
                cursor = await conn.cursor()
                if search_text:
                    await cursor.execute("""
                        SELECT id, main_url, nickname, video_url, "480p", "720p", "1080p", "1440p", "2160p", last_checked, status
                        FROM streams
                        WHERE lower(nickname) LIKE ?
                    """, (f"%{search_text}%",))
                else:
                    await cursor.execute("""
                        SELECT id, main_url, nickname, video_url, "480p", "720p", "1080p", "1440p", "2160p", last_checked, status
                        FROM streams
                    """)
                rows = await cursor.fetchall()
                self.db_tree.delete(*self.db_tree.get_children())
                for row in rows:
                    self.db_tree.insert("", "end", values=row)

        asyncio.run(async_search())

    def start_scraping(self):
        urls = self.url_entry.get().split()
        depth = self.depth_var.get()
        concurrency = self.concurrency_var.get()
        if not urls:
            messagebox.showerror("Ошибка", "Введите хотя бы один URL!")
            return
        if depth < 0 or depth > 3:
            messagebox.showerror("Ошибка", "Depth должен быть 0..3")
            return
        if concurrency < 1 or concurrency > 10:
            messagebox.showerror("Ошибка", "Concurrency 1..10")
            return

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.refresh_db_button.configure(state="disabled")
        self.progress_bar.start()
        self.clear_log()

        self.crawler_thread = threading.Thread(
            target=self.run_crawler,
            args=(urls, depth, concurrency)
        )
        self.crawler_thread.daemon = True
        self.crawler_thread.start()

    def run_crawler(self, urls, depth, concurrency):
        async def async_crawl():
            try:
                if len(urls) == 1:
                    crawler = SiteCrawler(urls[0], max_depth=depth, concurrency=concurrency, db_updated_callback=self.update_gui_db_table)
                    self.crawlers.append(crawler)
                    await crawler.crawl()
                else:
                    crs = [
                        SiteCrawler(u, max_depth=depth, concurrency=concurrency, db_updated_callback=self.update_gui_db_table)
                        for u in urls
                    ]
                    self.crawlers.extend(crs)
                    await asyncio.gather(*(c.crawl() for c in crs))
            except Exception as e:
                logger.error(f"Ошибка скрапинга: {e}")
            finally:
                for c in self.crawlers:
                    if c.browser:
                        await c.browser.close()
                    if c.playwright:
                        await c.playwright.stop()
                self.crawlers.clear()
                self.master.after(0, self.scraping_finished)

        asyncio.run(async_crawl())

    def stop_scraping(self):
        for c in self.crawlers:
            while not c.to_visit.empty():
                try:
                    c.to_visit.get_nowait()
                    c.to_visit.task_done()
                except asyncio.QueueEmpty:
                    break
        self.crawlers.clear()
        self.scraping_finished()

    def scraping_finished(self):
        self.progress_bar.stop()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.refresh_db_button.configure(state="normal")
        messagebox.showinfo("Готово", "Скрапинг завершён!")

    def clear_log(self):
        self.log_area.configure(state='normal')
        self.log_area.delete('1.0', tk.END)
        self.log_area.configure(state='disabled')

    def save_log(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log Files", "*.log"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.log_area.get("1.0", tk.END))
                messagebox.showinfo("Сохранено", f"Лог сохранен в {file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить лог: {e}")

    # NEW: экспортим таблицу videos в нужном формате
    def export_videos(self):
        """
        Экспорт из таблицы videos в текстовый файл в формате:

        Name of site link
        Main link
        Source link
        # --------------------------------------------------------------------------
        ...
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not file_path:
            return
        try:
            # Считаем все строки из БД
            async def do_export():
                lines = []
                async with aiosqlite.connect("streams.db") as conn:
                    cursor = await conn.cursor()
                    await cursor.execute("SELECT site_name, main_link, source_link FROM videos ORDER BY id")
                    rows = await cursor.fetchall()
                    for row in rows:
                        site_name, main_link, source_link = row
                        # Формируем блок из трёх строк
                        lines.append(site_name)
                        lines.append(main_link)
                        lines.append(source_link)
                        lines.append("# --------------------------------------------------------------------------")
                # Запишем в файл
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
            asyncio.run(do_export())
            messagebox.showinfo("Успех", f"Экспортировано в {file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать: {e}")

# --------------------------------------------------------------------------
# Точка входа (GUI)
# --------------------------------------------------------------------------
def main():
    root = tk.Tk()
    gui = CrawlerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    # Запуск GUI
    main()
    # Или для консольного режима (закомментируйте строку выше):
    # asyncio.run(console_main())
