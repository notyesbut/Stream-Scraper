import asyncio
import os
import subprocess
import time
import logging
import winreg

import aiosqlite
import websockets
from playwright.async_api import async_playwright, TimeoutError, Page
from typing import Any, List, Set, Dict, Tuple, Union
import sys
import urllib.parse
import re
import sqlite3
from datetime import datetime, timedelta, timezone
import aiohttp
from bs4 import BeautifulSoup
import json
import xml.etree.ElementTree as ET
import m3u8
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from typing import List
import threading
import webbrowser

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

    # Определенные расширения для стримов
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
        "video/mp4", # Тут нужно быть аккуратным, так как mp4 может быть и не стримом
        "video/x-flv",
    ]
    # Используем startswith вместо in для большей точности
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
    Использует 'unicode_escape', но игнорирует/заменяет недопустимые последовательности,
    чтобы избежать ошибок вида "unknown extension ?R at position ...".
    """
    try:
        # Используем 'unicode_escape' c игнорированием ошибочных последовательностей.
        return s.encode("utf-8", errors="ignore").decode("unicode_escape", errors="ignore")
    except UnicodeDecodeError:
        # Если декодирование не удалось, возвращаем оригинальную строку
        return s


# Простой паттерн для нахождения возможных URL в тексте.
URL_PATTERN = re.compile(
    r'''(?i)\b((?:https?://|//)
    [a-z0-9\-._~:/?#[\]@!$&'()*+,;=%]+)''',
    re.VERBOSE
)


def deep_search_for_urls(obj: Any) -> List[str]:
    """
    Рекурсивно ищет URL в любом месте вложенного JSON (словарь, список, строка).
    Возвращает список найденных ссылок.
    """
    results = []
    if isinstance(obj, dict):
        for val in obj.values():
            results.extend(deep_search_for_urls(val))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(deep_search_for_urls(item))
    elif isinstance(obj, str):
        # Ищем все URL внутри строки
        found = URL_PATTERN.findall(obj)
        results.extend(found)
    return results


# --------------------------------------------------------------------------
# Классы для парсинга плейлистов
# --------------------------------------------------------------------------

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

        # Парсинг сегментов
        for segment in self.playlist.segments:
            segment_uri = resolve_url(self.base_url, segment.uri)
            info["segments"].append({
                "uri": segment_uri,
                "duration": segment.duration,
                "title": segment.title,
                "byterange": segment.byterange
            })

        # Парсинг мастер-плейлистов (варианты качества)
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
        """
        Извлекает пространства имен из корневого элемента XML.
        """
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

        # Парсинг AdaptationSet
        for adaptation in self.root.findall('.//{*}AdaptationSet', namespaces=self.ns):
            adaptation_info = {
                "id": adaptation.attrib.get('id'),
                "mime_type": adaptation.attrib.get('mimeType'),
                "codecs": adaptation.attrib.get('codecs'),
                "segment_alignment": adaptation.attrib.get('segmentAlignment'),
                "representations": []
            }

            # Парсинг Representation
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

                # Построение полного URL для BaseURL, если он существует
                if rep_info["base_url"]:
                    rep_info["base_url"] = resolve_url(self.base_url, rep_info["base_url"])

                # Определение качества на основе разрешения
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
# Функции для работы с плейлистами
# --------------------------------------------------------------------------

async def fetch_content(session: aiohttp.ClientSession, url: str) -> Union[str, None]:
    """
    Асинхронно загружает содержимое по URL.
    """
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


async def analyze_playlist(url: str) -> Union[Dict[str, Any], None]:
    """
    Анализирует плейлист по заданному URL и возвращает информацию о нем.
    """
    cleaned_url = clean_url(url)
    playlist_type = determine_playlist_type(cleaned_url)

    if playlist_type == 'unknown':
        logger.warning(f"[analyze_playlist] Неизвестный тип плейлиста для URL: {cleaned_url}")
        return None

    async with aiohttp.ClientSession() as session:
        content = await fetch_content(session, cleaned_url)
        if not content:
            return None

        if playlist_type == 'hls':
            parser = HLSPlaylistParser(content, base_url=cleaned_url)
            info = parser.parse()
            return info

        elif playlist_type == 'dash':
            parser = DASHPlaylistParser(content, base_url=cleaned_url)
            info = parser.parse()
            return info

    return None


def determine_playlist_type(url: str) -> str:
    """
    Определяет тип плейлиста по расширению.
    Возвращает 'hls' для .m3u8, 'dash' для .mpd, иначе 'unknown'.
    """
    if url.lower().endswith('.m3u8'):
        return 'hls'
    elif url.lower().endswith('.mpd'):
        return 'dash'
    else:
        return 'unknown'


def resolve_url(base: str, link: str) -> str:
    """
    Преобразует относительный URL в абсолютный на основе базового URL.
    """
    return urllib.parse.urljoin(base, link)


# --------------------------------------------------------------------------
# Функции для работы с базой данных
# --------------------------------------------------------------------------

def init_db(db_name: str = "streams.db", qualities: List[str] = None):
    if qualities is None:
        qualities = ["480p", "720p", "1080p", "1440p", "2160p"]

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Создание таблицы streams с экранированными именами столбцов
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS streams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            main_url TEXT NOT NULL UNIQUE,
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
    logger.info("[Database] Таблица 'streams' создана или уже существует.")

    # Получаем существующие колонки
    cursor.execute("PRAGMA table_info(streams)")
    existing_columns = [info[1] for info in cursor.fetchall()]

    # Добавление столбцов для каждого качества, если они не существуют
    for quality in qualities:
        if quality not in existing_columns:
            try:
                cursor.execute(f'ALTER TABLE streams ADD COLUMN "{quality}" TEXT')
                logger.info(f"[DB] Добавлен столбец для качества: {quality}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    logger.debug(f"[DB] Столбец для качества '{quality}' уже существует.")
                else:
                    logger.error(f"[DB] Ошибка при добавлении столбца '{quality}': {e}")

    conn.commit()
    conn.close()
    logger.info(f"[Database] Инициализация базы данных завершена: {db_name}")


def update_stream_quality(db_name: str, stream_id: int, quality: str, url: str):
    """
    Обновляет запись в таблице streams, устанавливая ссылку для указанного качества.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    try:
        cursor.execute(f"""
            UPDATE streams
            SET `{quality}` = ?
            WHERE id = ?
        """, (url, stream_id))
        if cursor.rowcount == 0:
            logger.warning(f"[DB] Стрим с ID {stream_id} не найден для обновления.")
        else:
            logger.info(f"[DB] Обновлен stream ID {stream_id}: {quality} -> {url}")
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"[DB] Ошибка при обновлении stream ID {stream_id}: {e}")
    finally:
        conn.close()


def get_stream_by_main_url(db_name: str, main_url: str) -> Union[Dict[str, Any], None]:
    """
    Извлекает стрим по main_url из базы данных.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, video_url FROM streams
        WHERE main_url = ?
    """, (main_url,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return {"id": row[0], "video_url": row[1]}
    return None


def save_to_db(original: str, sources: Set[str], db_name: str = "streams.db"):
    """
    Сохраняет найденные стрим-ссылки в базу данных.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    for source in sources:
        try:
            # Получаем существующую запись по main_url
            cursor.execute("""
                SELECT id, video_url FROM streams
                WHERE main_url = ?
            """, (original,))
            row = cursor.fetchone()

            if row:
                stream_id, existing_video_url = row
                if source != existing_video_url:
                    # Обновляем video_url и другие поля
                    cursor.execute("""
                        UPDATE streams
                        SET video_url = ?, last_checked = ?, status = 'active'
                        WHERE id = ?
                    """, (source, datetime.now(timezone.utc), stream_id))
                    logger.info(f"[Database] Обновлена ссылка для {original}: {source}")
            else:
                # Вставляем новую запись
                cursor.execute("""
                    INSERT INTO streams (main_url, video_url, last_checked, status)
                    VALUES (?, ?, ?, 'active')
                """, (original, source, datetime.now(timezone.utc)))
                logger.info(f"[Database] Добавлена новая ссылка: {source}")
        except sqlite3.IntegrityError:
            logger.warning(f"[Database] Запись для {original} уже существует.")
        except sqlite3.Error as e:
            logger.error(f"[Database] Ошибка при вставке данных: {e}")
    conn.commit()
    conn.close()
    logger.info(f"[DB] Сохранены стрим-ссылки для: {original}")


# --------------------------------------------------------------------------
# Класс NetworkWatcher для отслеживания сетевых запросов
# --------------------------------------------------------------------------
class NetworkWatcher:
    def __init__(self):
        self.found_links: Set[str] = set()
        self.is_closed = False
        self.websocket_buffer: Dict[str, str] = {}  # Буфер для данных из WebSocket

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
            logger.error(f"[NetworkWatcher] Ошибка в on_request_finished: {e}")

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

                elif opcode == 8: # Закрытие соединения
                    await self.on_websocket_close(ws)

            elif isinstance(frame, str):
                logger.debug(f"Получено строковое сообщение: {frame}")
                self.websocket_buffer[ws.url] += frame

            else:
                logger.warning(f"Неожиданный тип фрейма: {type(frame)}")

        except (ConnectionResetError, websockets.exceptions.ConnectionClosed) as e:
            logger.error(f"[NetworkWatcher/WebSocket] Ошибка соединения: {e}")

    async def on_websocket_close(self, ws):
        if self.is_closed: return
        if ws.url in self.websocket_buffer:
          data = self.websocket_buffer[ws.url]
          logger.debug(f"[NetworkWatcher/WebSocket] Закрыто соединение, анализ буфера: {ws.url}")
          found = URL_PATTERN.findall(data)
          for url_ in found:
              await self._process_potential_stream(url_)
          del self.websocket_buffer[ws.url]

    def close(self):
        self.is_closed = True
        self.websocket_buffer.clear()

# --------------------------------------------------------------------------
# Класс PageParser для парсинга содержимого страницы
# --------------------------------------------------------------------------
class PageParser:
    """
    Собирает title, ссылки в <video>, <source>, а также ищет <script> + JSON.
    """

    def __init__(self, page: Page):
        self.page = page
        self.results: Dict[str, Any] = {
            "page_title": None,
            "videos": [],
            "sources": [],
            "script_urls": []
        }

    async def parse(self) -> Dict[str, Any]:
        """Запускает парсинг главного фрейма и всех iframes."""
        # Парсинг главного фрейма
        await self._parse_frame(self.page.main_frame)

        # Парсинг всех фреймов, кроме главного
        frames = self.page.frames
        other_frames = [fr for fr in frames if fr != self.page.main_frame]
        if other_frames:
          await asyncio.gather(*[self._parse_frame(fr) for fr in other_frames])

        return self.results

    async def _parse_frame(self, frame):
        """Парсит содержимое одного фрейма (HTML, video/source, script)."""
        try:
            html = await frame.content()
        except Exception as e:
            logger.error(f"[PageParser] Ошибка при получении содержимого фрейма: {e}")
            return

        soup = BeautifulSoup(decode_js_escapes(html), "html.parser")

        # Если это главный фрейм, ищем <title>
        if frame == self.page.main_frame:
            title_tag = soup.find("title")
            if title_tag:
                self.results["page_title"] = title_tag.get_text(strip=True)
                logger.info(f"[PageParser] Заголовок страницы: {self.results['page_title']}")

        # Ищем видео и источники
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
                    logger.info(f"[PageParser] Найдена source-ссылка: {cleaned}")

        # Ищем <script>, парсим URL и пробуем распознать JSON
        for sc in soup.find_all("script"):
            stxt = decode_js_escapes(sc.get_text() or "").strip()

            # Эвристика для поиска JSON с потенциальными стримами
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
                                logger.info(f"[PageParser] Найдена стрим-ссылка из JSON: {cleaned}")
                    except json.JSONDecodeError:
                        logger.debug("[PageParser] Скрипт не является валидным JSON — пропущен.")

            # Ищем любые URL с помощью регулярного выражения
            found = URL_PATTERN.findall(stxt)
            for url_ in found:
                cleaned = clean_url(url_)
                if is_stream_url(cleaned, "") and not is_placeholder_link(cleaned):
                    self.results["script_urls"].append(cleaned)
                    logger.info(f"[PageParser] Найдена стрим-ссылка из скрипта: {cleaned}")

# --------------------------------------------------------------------------
# Функция для клика по кнопкам "Play" и запуска <video>.play()
# --------------------------------------------------------------------------
async def click_play_buttons(page: Page):
    """
    Пытается кликнуть по разным селекторам Play,
    а также вызывает JS для запуска video.play().
    """
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
        "video" # Добавлено так как часто на видео нужно кликать чтобы запустить
    ]

    clicked = False
    for sel in play_selectors:
        try:
            await page.click(sel, timeout=500)
            logger.debug(f"[click_play_buttons] Clicked {sel} on main page")
            clicked = True
            break  # Выходим из цикла, если клик успешен
        except Exception:
            logger.debug(f"[click_play_buttons] Не удалось кликнуть: {sel}")
            continue

    # Если клик на главной странице был успешным, не нужно кликать на фреймах
    if not clicked:
        # Клик на <video> во вложенных фреймах
        for fr in page.frames:
            if fr != page.main_frame:
                try:
                    await fr.click("video", timeout=500)
                    logger.debug("[click_play_buttons] Clicked <video> in iframe")
                    break
                except Exception:
                    logger.debug("[click_play_buttons] Не удалось кликнуть <video> во фрейме")
                    continue

    # Пробуем запустить видео через JS
    try:
        await page.evaluate(
            "() => { document.querySelectorAll('video').forEach(v => v.play().catch(()=>{})) }"
        )
        logger.debug("[click_play_buttons] Called video.play() on all <video> elements.")
    except Exception as e:
        logger.error(f"[click_play_buttons] Ошибка при вызове video.play(): {e}")

    # Эмуляция пользовательского поведения: прокрутка, движение мыши (Можно убрать эти две строки, если не нужно)
    try:
        await page.mouse.move(100, 100)
        await page.mouse.wheel(0, 500)
        logger.debug("[click_play_buttons] Emulated mouse move & scroll.")
    except Exception as e:
        logger.error(f"[click_play_buttons] Ошибка эмуляции действий: {e}")

# --------------------------------------------------------------------------
# Функция для отображения результатов
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
    print("\n--- Детали парсинга ---")
    if results.get("videos"):
        print("Видео-ссылки:", results.get("videos", []))
    if results.get("sources"):
        print("Source-ссылки:", results.get("sources", []))
    if results.get("script_urls"):
        print("Script-ссылки:", results.get("script_urls", []))
    print("-" * 30, "\n")


# --------------------------------------------------------------------------
# Класс SiteCrawler для управления процессом кроулинга
# --------------------------------------------------------------------------
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
        self.playwright = None  # Добавляем экземпляр Playwright
        self.browser = None  # Добавляем экземпляр браузера
        self.db_updated_callback = db_updated_callback  # Сохраняем callback


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
        self.init_database()

        URL_PATTERN = re.compile(
            r'''(?i)\b((?:https?://|//)
            [a-z0-9\-._~:/?#[\]@!$&'()*+,;=%]+)''',
            re.VERBOSE
        )

    def init_database(self):
        qualities = ["480p", "720p", "1080p", "1440p", "2160p"]
        init_db(self.db_name, qualities=qualities)

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
                        await element.click(timeout=5000)
                        logger.info(f"[Popup Handler] Выполнено '{action}' => '{selector}'")
                    elif action == "accept":
                        await element.click(timeout=5000)
                        logger.info(f"[Popup Handler] Выполнено '{action}' => '{selector}'")
            except Exception as e:
                logger.debug(f"[Popup Handler] Не удалось обработать '{selector}': {e}")

    async def click_preview_images(self, page: Page):
        clicked_images: Set[str] = set()  # Множество для отслеживания кликнутых изображений
        try:
            preview_images = await page.query_selector_all("img[src]")
            for img in preview_images:
                src = await img.get_attribute("src")
                if src and self.preview_image_pattern.match(src) and src not in clicked_images:
                    try:
                        await img.click(timeout=5000)
                        clicked_images.add(src)
                        logger.info(f"[Preview Clicker] Кликнуто по превью: {src}")
                        await page.wait_for_load_state("networkidle", timeout=5000)
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"[Preview Clicker] Ошибка клика по превью {src}: {e}")
        except Exception as e:
            logger.error(f"[Preview Clicker] Ошибка поиска превью: {e}")

    async def extract_internal_links(self, page: Page) -> Set[str]:
        internal_links = set()
        try:
            content_divs = await page.query_selector_all(f"div.{self.content_class}")
            for content_div in content_divs:
                link_elements = await content_div.query_selector_all("a[href]")
                for link in link_elements:
                    href = await link.get_attribute("href")
                    if href:
                        absolute_url = urllib.parse.urljoin(self.start_url, href)
                        absolute_url = clean_url(absolute_url)
                        if self.is_same_domain(self.start_url, absolute_url):
                            internal_links.add(absolute_url)
            logger.info(f"[Crawler] Найдено {len(internal_links)} внутренних ссылок внутри '{self.content_class}' на странице: {page.url}")
        except Exception as e:
            logger.error(f"[Crawler] Ошибка при извлечении внутренних ссылок: {e}")
        return internal_links

    async def update_database(self, main_url: str, video_urls: Set[str]):
        """Обновляет базу данных, добавляя или обновляя информацию о стримах."""
        async with aiosqlite.connect(self.db_name) as conn:
            async with conn.cursor() as cursor:
                try:
                    if video_urls:
                        for url in video_urls:
                            await cursor.execute("""
                                INSERT INTO streams (main_url, video_url, last_checked, status, success)
                                VALUES (?, ?, ?, 'active', TRUE)
                                ON CONFLICT(main_url) DO UPDATE SET
                                    video_url = excluded.video_url,
                                    last_checked = excluded.last_checked,
                                    status = excluded.status,
                                    success = TRUE
                            """, (main_url, url, datetime.now(timezone.utc)))
                            logger.info(f"[Database] Обновлена/добавлена ссылка для {main_url}: {url}")
                        await conn.commit()
                    logger.info(f"[Database] Обновление завершено: {main_url}")

                    # Сигнализируем об обновлении БД
                    if self.db_updated_callback:
                        self.db_updated_callback()

                except sqlite3.Error as e:
                    logger.error(f"[Database] Ошибка обновления: {e}")

    async def crawl_url(self, page: Page, current_url: str, depth: int):
        """Обрабатывает URL, собирает ссылки, данные и обновляет БД."""
        try:
            logger.info(f"[Crawler] Обработка URL: {current_url} (Глубина: {depth})")
            self.visited_urls.add(current_url)

            for attempt in range(self.max_retries):
                try:
                    await page.goto(current_url, timeout=30000)
                    break  # Успешно загрузили - выходим из цикла
                except TimeoutError:
                    logger.warning(
                        f"[Crawler] Таймаут при загрузке {current_url}, попытка {attempt + 1}/{self.max_retries}"
                    )
                    if attempt == self.max_retries - 1:
                        raise  # Если последняя попытка - выбрасываем исключение

            await self.handle_popups(page)
            internal_links = await self.extract_internal_links(page)
            for link in internal_links:
                if link not in self.visited_urls:
                    # Определение базового пути для предотвращения зацикливания
                    base_path = urllib.parse.urlparse(link).path.split('/')[1]  # Берем первый сегмент пути
                    if base_path:
                        if base_path not in self.url_visit_counts:
                            self.url_visit_counts[base_path] = 0
                        self.url_visit_counts[base_path] += 1

                        # Проверка на превышение лимита посещений для базового пути
                        if self.url_visit_counts[base_path] <= self.loop_prevention_threshold:
                            await self.to_visit.put((link, depth + 1))
                        else:
                            logger.debug(f"[Crawler] Превышен лимит посещений для {base_path}: {link}")

            await self.click_preview_images(page)
            await self.handle_popups(page)

            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
                logger.info("[Crawler] networkidle достигнуто.")
            except TimeoutError:
                logger.warning("[Crawler] Превышено время ожидания networkidle.")

            parser = PageParser(page)
            nw = NetworkWatcher()
            page.on("response", lambda r: asyncio.create_task(nw.on_response(r)))
            page.on("requestfinished", lambda r: asyncio.create_task(nw.on_request_finished(r)))
            page.on("websocket", nw.on_websocket)
            page.on("request", nw.on_request)

            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                logger.info(f"[Crawler] Попытка #{attempt} Play.")
                await click_play_buttons(page)
                await asyncio.sleep(2)

                current_results = await parser.parse()

                video_urls = set()
                for candidate in (
                        current_results["videos"] +
                        current_results["sources"] +
                        current_results["script_urls"]
                ):
                    cand_clean = clean_url(decode_js_escapes(candidate))
                    if is_stream_url(cand_clean, "") and not is_placeholder_link(cand_clean):
                        if await is_url_valid(cand_clean):
                            video_urls.add(cand_clean)

                video_urls.update(nw.found_links)

                if video_urls:
                    logger.info("[Crawler] Потоки найдены, завершаем попытки.")
                    break
                else:
                    logger.info("[Crawler] Потоки не найдены, продолжаем...")

            await self.update_database(current_url, video_urls)

            parse_results = {
                "page_title": current_results.get("page_title"),
                "videos": list(video_urls),
                "sources": current_results.get("sources", []),
                "script_urls": current_results.get("script_urls", [])
            }
            display_results(current_url, video_urls, parse_results)

            await self.process_streams(video_urls, current_url)

        except Exception as e:
            logger.error(f"[Crawler] Ошибка => {current_url}: {e}")
        finally:
            await page.close()

    async def process_streams(self, video_urls: Set[str], main_url: str):
        """Обрабатывает найденные стрим-ссылки."""
        async with aiohttp.ClientSession() as session:  # Добавляем сессию aiohttp
            tasks = [self.process_stream(stream_url, main_url, session) for stream_url in video_urls]
            await asyncio.gather(*tasks)

    async def process_stream(self, stream_url: str, main_url: str, session: aiohttp.ClientSession):
        """Анализирует плейлист и обновляет БД."""
        try:
            info = await analyze_playlist(stream_url)
            if not info:
                logger.warning(f"[Process Stream] Не удалось проанализировать: {stream_url}")
                return
            extracted_qualities = {}
            if info["type"] == "HLS":
                for pl in info.get("playlists", []):
                    quality = pl.get("quality")
                    uri = pl.get("uri")
                    if quality and uri:
                        extracted_qualities[quality] = uri
            elif info["type"] == "DASH":
                for adap_set in info.get("adaptation_sets", []):
                    for rep in adap_set.get("representations", []):
                        quality = rep.get("quality")
                        media_url = rep.get("segment_template", {}).get("media")
                        if quality and media_url:
                            extracted_qualities[quality] = media_url

            async with aiosqlite.connect(self.db_name) as conn:  # Открываем соединение с БД
                async with conn.cursor() as cursor:
                    for q, url_ in extracted_qualities.items():
                        if q in ["480p", "720p", "1080p", "1440p", "2160p"]:
                            # Обновляем запись в БД, добавляя URL для соответствующего качества
                            await cursor.execute(f"""
                                UPDATE streams
                                SET `{q}` = ?
                                WHERE main_url = ?
                            """, (url_, main_url))
                            if cursor.rowcount > 0:
                                logger.info(f"[DB] Обновлена ссылка для качества {q} стрима {main_url}: {url_}")

                    await conn.commit()
                    if self.db_updated_callback:
                        self.db_updated_callback()

        except Exception as e:
            logger.error(f"[Process Stream] Ошибка при обработке стрима {stream_url}: {e}")


    def update_stream_quality_in_db(self, main_url: str, stream_url: str, quality: str, url: str):
        """Обновляет информацию о качестве стрима в БД."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        try:
            cursor.execute(f"""
                UPDATE streams
                SET `{quality}` = ?
                WHERE main_url = ? AND video_url = ?
            """, (url, main_url, stream_url))
            if cursor.rowcount == 0:
                logger.warning(f"[DB] Стрим не найден: {stream_url}")
            else:
                logger.info(f"[DB] Обновлен стрим {stream_url} => {quality} => {url}")
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"[DB] Ошибка обновления стрима: {stream_url}, {quality}: {e}")
        finally:
            conn.close()

    async def crawl(self):
        """Запускает процесс кроулинга."""
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
        """Задача-воркер для обработки URL из очереди."""
        while True:
            try:
                current_url, depth = await self.to_visit.get()
            except asyncio.QueueEmpty:
                if not await self.check_inactive_streams():
                    break

            if current_url in self.visited_urls or self.should_exclude(current_url) or depth > self.max_depth:
                self.to_visit.task_done()
                continue

            # Ограничиваем количество посещенных URL
            if len(self.visited_urls) >= self.visited_urls_limit:
                logger.warning(f"[Crawler] Достигнут лимит посещенных URL: {self.visited_urls_limit}")
                break

            # убрал async with self.global_semaphore так как эта логика уже управляется ограничением в пуле потоков
            await self.semaphore.acquire()
            try:
                await self.process_url(context, current_url, depth)
            finally:
                self.semaphore.release()
                self.to_visit.task_done()

    async def process_url(self, context, current_url: str, depth: int):
        """Создает новую страницу и запускает процесс обхода для заданного URL."""
        page = await context.new_page()
        await self.crawl_url(page, current_url, depth)

    def check_inactive_streams(self):  # <- убрал async
        """Периодически проверяет неактивные стримы и добавляет их в очередь."""
        conn = sqlite3.connect(self.db_name)  # <- синхронное подключение
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT main_url
                FROM streams
                WHERE status = 'inactive' AND success = TRUE AND last_checked < ?
            """, (datetime.now(timezone.utc) - timedelta(seconds=self.inactive_check_interval),))
            rows = cursor.fetchall()
            if rows:
                for row in rows:
                    main_url = row[0]
                    self.to_visit.put_nowait((main_url, 0))  # Добавляем в очередь без await
                logger.info(f"[Crawler] Добавлено {len(rows)} неактивных стримов для перепроверки.")
                return True
        except sqlite3.Error as e:
            logger.error(f"[Crawler] Ошибка при проверке неактивных стримов: {e}")
        finally:
            conn.close()
        return False

        return False

    async def update_and_check_url(self, url, context):
        """Обновляет и проверяет URL."""
        try:
            page = await context.new_page()
            nw = NetworkWatcher()
            page.on("response", lambda r: asyncio.create_task(nw.on_response(r)))
            page.on("requestfinished", lambda r: asyncio.create_task(nw.on_request_finished(r)))
            page.on("websocket", nw.on_websocket)
            page.on("request", nw.on_request)
            await self.crawl_url(page, url, 0)  # Используем crawl_url с глубиной 0
            await page.close()
        except Exception as e:
            logger.error(f"Ошибка при обновлении и проверке {url}: {e}")


# --------------------------------------------------------------------------
# Функция для проверки валидности URL
# --------------------------------------------------------------------------
async def is_url_valid(url: str) -> bool:
    """
    Проверяет доступность URL путем отправки HEAD-запроса.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=10) as resp:
                return resp.status == 200
    except Exception:
        return False


# --------------------------------------------------------------------------
# Основная функция для запуска кроулинга
# --------------------------------------------------------------------------
async def main():
    # Инициализируем базу данных
    init_db()

    urls = []

    # Если переданы аргументы командной строки, используем их
    if len(sys.argv) > 1:
        urls = sys.argv[1:]
    else:
        # Если нет аргументов, запрашиваем ввод
        print("Введите ссылки (разделяйте пробелами):")
        user_input = input()
        urls = user_input.strip().split()

    if not urls:
        print("Нет URL, завершение.")
        return

    # Если один сайт
    if len(urls) == 1:
        site = urls[0]
        crawler = SiteCrawler(
            start_url=site,
            max_depth=3,
            exclude_patterns=[
                "settings", "swag", "terms", "privacy", "conditions", "support",
                "dmca", "remove-content", "feedback", "security-center",
                "law-enforcement", "report", "nonconsensual", "abusive-content",
                "billing", "disable-account", "apps", "contest", "affiliates",
                "jobs", "sitemap", "language"
            ],
            db_name="streams.db",
            popup_selectors=[
                {"selector": "button.close, button.close-button, .modal-close, .close-modal", "action": "click"},
                {"selector": "button.accept, button.agree, button#accept, button#agree", "action": "click"},
                {"selector": "button.confirm, button#confirm", "action": "click"},
                {"selector": "button#age-confirm, button#yes", "action": "click"},
            ],
            preview_image_pattern=r"https://thumb\.live\.mmcdn\.com/.*\.png",
            content_class="content",
            concurrency=5
        )
        await crawler.crawl()
    else:
        logger.info("[main] Параллельный кроулинг нескольких сайтов.")
        crawlers = [
            SiteCrawler(
                start_url=site,
                max_depth=3,
                exclude_patterns=[
                    "settings", "swag", "terms", "privacy", "conditions", "support",
                    "dmca", "remove-content", "feedback", "security-center",
                    "law-enforcement", "report", "nonconsensual", "abusive-content",
                    "billing", "disable-account", "apps", "contest", "affiliates",
                    "jobs", "sitemap", "language"
                ],
                db_name="streams.db",
                popup_selectors=[
                    {"selector": "button.close, button.close-button, .modal-close, .close-modal", "action": "click"},
                    {"selector": "button.accept, button.agree, button#accept, button#agree", "action": "click"},
                    {"selector": "button.confirm, button#confirm", "action": "click"},
                    {"selector": "button#age-confirm, button#yes", "action": "click"},
                ],
                preview_image_pattern=r"https://thumb\.live\.mmcdn\.com/.*\.png",
                content_class="content",
                concurrency=5
            ) for site in urls
        ]
        await asyncio.gather(*(crawler.crawl() for crawler in crawlers))


class CrawlerGUI:
    def __init__(self, master):
        self.vlc_path = tk.StringVar(value=self.find_vlc_path())  # Путь к VLC, если найден

        self.master = master
        master.title("Стрим-Скрапер")

        # Настройка стиля
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat")
        style.configure("Treeview", rowheight=25) # Увеличиваем высоту строк в Treeview

        # --- Вкладки ---
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill="both", expand=True)

        self.crawl_frame = ttk.Frame(self.notebook)
        self.db_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.crawl_frame, text="Scraping")
        self.notebook.add(self.db_frame, text="Database")

        # --- Переменные для управления процессом ---
        self.loop = asyncio.get_event_loop()
        self.crawler_thread = None
        self.crawlers: List[SiteCrawler] = []

        # --- Виджеты для вкладки "Скрапинг" ---
        self.create_crawl_widgets()

        # --- Виджеты для вкладки "База данных" ---
        self.create_db_widgets()

        # --- Инициализация базы данных ---
        init_db()

        # --- Загрузка данных в таблицу ---
        self.load_db_data()

    def create_crawl_widgets(self):
        """Создает виджеты для вкладки "Скрапинг"."""

        self.refresh_db_button = ttk.Button(self.crawl_frame, text="Refresh DataBase", command=self.refresh_db)
        self.refresh_db_button.grid(row=2, column=4, padx=5, pady=5, sticky="ew")

        # --- Ввод URL ---
        self.url_label = ttk.Label(self.crawl_frame, text="URL (link (space) link):")
        self.url_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.url_entry = ttk.Entry(self.crawl_frame, width=50)
        self.url_entry.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
        self.url_entry.insert(0, "https://www.google.com")  # Пример URL

        # --- Настройки VLC ---
        self.vlc_path_label = ttk.Label(self.crawl_frame, text="Path VLC:")
        self.vlc_path_label.grid(row=6, column=0, sticky="w", padx=5, pady=5)

        self.vlc_path_entry = ttk.Entry(self.crawl_frame, textvariable=self.vlc_path, width=40)
        self.vlc_path_entry.grid(row=6, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

        self.vlc_path_button = ttk.Button(self.crawl_frame, text="Path", command=self.browse_vlc_path)
        self.vlc_path_button.grid(row=6, column=3, padx=5, pady=5, sticky="ew")


        # --- Глубина поиска ---
        self.depth_label = ttk.Label(self.crawl_frame, text="Deepth (0 - 3):")
        self.depth_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)

        self.depth_var = tk.IntVar(value=3)
        self.depth_entry = ttk.Entry(self.crawl_frame, textvariable=self.depth_var, width=5)
        self.depth_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # --- Параллельные задачи ---
        self.concurrency_label = ttk.Label(self.crawl_frame, text="Thread count (1 - 5):")
        self.concurrency_label.grid(row=1, column=2, sticky="w", padx=5, pady=5)

        self.concurrency_var = tk.IntVar(value=5)
        self.concurrency_entry = ttk.Entry(self.crawl_frame, textvariable=self.concurrency_var, width=5)
        self.concurrency_entry.grid(row=1, column=3, sticky="w", padx=5, pady=5)

        # --- Кнопки ---
        self.start_button = ttk.Button(self.crawl_frame, text="Start", command=self.start_scraping)
        self.start_button.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        self.stop_button = ttk.Button(self.crawl_frame, text="Stop", command=self.stop_scraping, state="disabled")
        self.stop_button.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.clear_log_button = ttk.Button(self.crawl_frame, text="Clear logs", command=self.clear_log)
        self.clear_log_button.grid(row=2, column=2, padx=5, pady=5, sticky="ew")

        self.save_log_button = ttk.Button(self.crawl_frame, text="Save logs", command=self.save_log)
        self.save_log_button.grid(row=2, column=3, padx=5, pady=5, sticky="ew")

        # --- Лог ---
        self.log_label = ttk.Label(self.crawl_frame, text="Log:")
        self.log_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)

        self.log_area = scrolledtext.ScrolledText(self.crawl_frame, wrap=tk.WORD, height=20)
        self.log_area.grid(row=4, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")
        self.log_area.configure(state='disabled')

        # --- Прогресс ---
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.crawl_frame, orient="horizontal", length=300, mode="indeterminate", variable=self.progress_var)
        self.progress_bar.grid(row=5, column=0, columnspan=4, padx=5, pady=10, sticky="ew")

        # --- Конфигурация колонок и строк ---
        self.crawl_frame.columnconfigure(1, weight=1)
        self.crawl_frame.rowconfigure(4, weight=1)

        # --- Перенаправление stdout и stderr в ScrolledText ---
        sys.stdout = TextRedirector(self.log_area, "stdout")
        sys.stderr = TextRedirector(self.log_area, "stderr")

    def browse_vlc_path(self):
        """Открывает диалог выбора файла для указания пути к VLC."""
        file_path = filedialog.askopenfilename(
            initialdir=".",
            title="select vlc.exe",
            filetypes=(("Executable files", "*.exe"), ("All files", "*.*"))
        )
        if file_path:
            self.vlc_path.set(file_path)

    def find_vlc_path(self):
        """Пытается автоматически найти путь к VLC."""
        vlc_path = None
        if sys.platform == "win32":
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\VideoLAN\VLC") as key:
                    vlc_path = winreg.QueryValueEx(key, "InstallDir")[0] + "\\vlc.exe"
            except FileNotFoundError:
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Wow6432Node\VideoLAN\VLC") as key:
                        vlc_path = winreg.QueryValueEx(key, "InstallDir")[0] + "\\vlc.exe"
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
        """Создает виджеты для вкладки "База данных"."""
        # --- Таблица ---
        self.db_tree = ttk.Treeview(self.db_frame, columns=("id", "main_url", "video_url", "480p", "720p", "1080p", "1440p", "2160p", "last_checked", "status"), show="headings")
        self.db_tree.pack(fill="both", expand=True)

        # --- Колонки ---
        self.db_tree.heading("id", text="ID")
        self.db_tree.heading("main_url", text="Main URL")
        self.db_tree.heading("video_url", text="Video URL")
        self.db_tree.heading("480p", text="480p")
        self.db_tree.heading("720p", text="720p")
        self.db_tree.heading("1080p", text="1080p")
        self.db_tree.heading("1440p", text="1440p")
        self.db_tree.heading("2160p", text="2160p")
        self.db_tree.heading("last_checked", text="Last Checked")
        self.db_tree.heading("status", text="Status")

        self.db_tree.column("id", width=30, anchor="center")
        self.db_tree.column("main_url", anchor="w")
        self.db_tree.column("video_url", anchor="w")
        self.db_tree.column("480p", anchor="w")
        self.db_tree.column("720p", anchor="w")
        self.db_tree.column("1080p", anchor="w")
        self.db_tree.column("1440p", anchor="w")
        self.db_tree.column("2160p", anchor="w")
        self.db_tree.column("last_checked", width=120, anchor="center")
        self.db_tree.column("status", width=80, anchor="center")

        # --- Контекстное меню ---
        self.db_tree.bind("<Button-3>", self.show_context_menu)

        # --- Сортировка ---
        for col in ("id", "main_url", "video_url", "480p", "720p", "1080p", "1440p", "2160p", "last_checked", "status"):
            self.db_tree.heading(col, text=col, command=lambda c=col: self.sort_column(c, False))

    def show_context_menu(self, event):
        """Отображает контекстное меню."""
        region = self.db_tree.identify_region(event.x, event.y)
        if region == "cell":
            item = self.db_tree.identify_row(event.y)
            column = self.db_tree.identify_column(event.x)
            col_id = int(column[1:]) - 1

            if item and column:
                self.db_tree.selection_set(item)
                self.context_menu = tk.Menu(self.master, tearoff=0)

                # Проверка, является ли содержимое ячейки ссылкой
                cell_value = self.db_tree.item(item, "values")[col_id]

                if cell_value and cell_value.startswith("http"):
                  self.context_menu.add_command(label="Open in VLC", command=lambda: self.open_in_vlc(cell_value))
                  self.context_menu.add_command(label="Open in browser", command=lambda: self.open_in_browser(cell_value))
                self.context_menu.add_command(label="copy", command=lambda i=item, c=col_id: self.copy_cell(i, c))

                # Для редактирования нужны права админа
                # self.context_menu.add_command(label="Изменить", command=lambda i=item, c=column: self.edit_cell(i, c))

                try:
                    self.context_menu.post(event.x_root, event.y_root)
                finally:
                    self.context_menu.grab_release()

    def open_in_vlc(self, url):
        """Открывает URL в VLC."""
        try:
            # Пытаемся найти путь к VLC в реестре Windows (для разных версий)
            vlc_path = None
            if sys.platform == "win32":
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\VideoLAN\VLC") as key:
                        vlc_path = winreg.QueryValueEx(key, "InstallDir")[0] + "\\vlc.exe"
                except:
                    pass

                if not vlc_path:
                    try:
                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Wow6432Node\VideoLAN\VLC") as key:
                            vlc_path = winreg.QueryValueEx(key, "InstallDir")[0] + "\\vlc.exe"
                    except:
                        pass

                if not vlc_path:
                    # Если не нашли в реестре, проверяем пути по умолчанию
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

            # Если нашли путь к VLC или это не Windows
            if vlc_path:
                subprocess.Popen([vlc_path, url])  # Запускаем VLC с URL
            else:
                # Если не нашли VLC, пытаемся открыть в браузере
                webbrowser.open(url)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open URL in VLC: {e}")

    def open_in_browser(self, url):
        """Открывает URL в браузере."""
        try:
            webbrowser.open(url)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open URL in browser: {e}")

    def refresh_db(self):
        """Запускает процесс обновления и проверки ссылок из БД."""
        if messagebox.askyesno("confirmation",
                               "Are you sure you want to update and check all links in the database? This may take a long time."):
            self.progress_bar.start()
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.refresh_db_button.configure(state="disabled")

            self.refresh_thread = threading.Thread(target=self.run_db_refresh)
            self.refresh_thread.daemon = True
            self.refresh_thread.start()

    def run_db_refresh(self):
        """Выполняет обновление и проверку ссылок в отдельном потоке."""

        async def update_database_async():
            """Обновляет базу данных асинхронно."""
            try:
                async with aiosqlite.connect("streams.db") as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("SELECT main_url FROM streams")
                        rows = await cursor.fetchall()

                urls_to_check = [row[0] for row in rows]

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

                    async def update_and_check_url(url):
                        try:
                            page = await context.new_page()
                            nw = NetworkWatcher()
                            page.on("response", lambda r: asyncio.create_task(nw.on_response(r)))
                            page.on("requestfinished", lambda r: asyncio.create_task(nw.on_request_finished(r)))
                            page.on("websocket", nw.on_websocket)
                            page.on("request", nw.on_request)

                            # Создание экземпляра SiteCrawler с нужными параметрами
                            crawler = SiteCrawler(
                                start_url=url,
                                max_depth=0,
                                db_name="streams.db",
                                concurrency=self.concurrency_var.get(),
                                exclude_patterns=[
                                    "settings", "swag", "terms", "privacy", "conditions", "support",
                                    "dmca", "remove-content", "feedback", "security-center",
                                    "law-enforcement", "report", "nonconsensual", "abusive-content",
                                    "billing", "disable-account", "apps", "contest", "affiliates",
                                    "jobs", "sitemap", "language"
                                ],
                                popup_selectors=[
                                    {"selector": "button.close, button.close-button, .modal-close, .close-modal",
                                     "action": "click"},
                                    {"selector": "button.accept, button.agree, button#accept, button#agree",
                                     "action": "click"},
                                    {"selector": "button.confirm, button#confirm", "action": "click"},
                                    {"selector": "button#age-confirm, button#yes", "action": "click"},
                                ],
                                preview_image_pattern=r"https://thumb\.live\.mmcdn\.com/.*\.png",
                                content_class="content"
                            )

                            await crawler.crawl_url(page, url, 0)  # Используем crawl_url с глубиной 0
                            await page.close()
                        except Exception as e:
                            logger.error(f"Ошибка при обновлении и проверке {url}: {e}")

                    tasks = [update_and_check_url(url) for url in urls_to_check]
                    await asyncio.gather(*tasks)
                    await context.close()
                    await browser.close()

            except Exception as e:
                logger.error(f"Ошибка при обновлении БД: {e}")
            finally:
                self.master.after(0, self.refresh_db_finished)

        threading.Thread(target=lambda: asyncio.run(update_database_async()), daemon=True).start()

    async def update_and_check_url(self, url, context):
        """Обновляет и проверяет URL."""
        try:
            page = await context.new_page()
            nw = NetworkWatcher()
            page.on("response", lambda r: asyncio.create_task(nw.on_response(r)))
            page.on("requestfinished", lambda r: asyncio.create_task(nw.on_request_finished(r)))
            page.on("websocket", nw.on_websocket)
            page.on("request", nw.on_request)

            # Создание экземпляра SiteCrawler с нужными параметрами
            crawler = SiteCrawler(
                start_url=url,
                max_depth=0,  # Устанавливаем глубину 0, так как обновляем только один URL
                db_name=self.db_name, # Используем имя базы данных из основного экземпляра
                concurrency=self.concurrency, # Используем настройки из основного экземпляра
                global_concurrency=self.global_concurrency, # Используем настройки из основного экземпляра
                exclude_patterns=self.exclude_patterns, # Используем настройки из основного экземпляра
                popup_selectors=self.popup_selectors, # Используем настройки из основного экземпляра
                preview_image_pattern=self.preview_image_pattern, # Используем настройки из основного экземпляра
                content_class=self.content_class # Используем настройки из основного экземпляра
            )

            await crawler.crawl_url(page, url, 0)  # Используем crawl_url с глубиной 0
            await page.close()
        except Exception as e:
            logger.error(f"Ошибка при обновлении и проверке {url}: {e}")

    def refresh_db_finished(self):
        """Действия после завершения обновления БД."""
        self.progress_bar.stop()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.refresh_db_button.configure(state="normal")
        self.update_gui_db_table()
        messagebox.showinfo("Готово", "Обновление БД завершено!")

    def update_gui_db_table(self):
        """Обновляет таблицу с данными из БД в GUI."""
        self.master.after(0, self._update_gui_db_table_task)


    def _update_gui_db_table_task(self):
        """Задача для обновления таблицы в главном потоке."""
        self.db_tree.delete(*self.db_tree.get_children())
        self.load_db_data()

    def copy_cell(self, item, col_id):
        """Копирует содержимое ячейки в буфер обмена."""
        try:
            cell_value = self.db_tree.item(item, "values")[col_id]
            self.master.clipboard_clear()
            self.master.clipboard_append(cell_value)
            messagebox.showinfo("Скопировано", "Содержимое ячейки скопировано в буфер обмена.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось скопировать: {e}")

    def edit_cell(self, item, column):
      """Позволяет редактировать содержимое ячейки."""

      def save_edit(event=None):
          """Сохраняет изменения в базе данных."""
          new_value = edit_entry.get()
          col_name = self.db_tree.heading(column, "text")
          item_id = self.db_tree.item(item, "values")[0]  # ID строки

          conn = sqlite3.connect("streams.db")
          cursor = conn.cursor()
          try:
              cursor.execute(f"UPDATE streams SET `{col_name}` = ? WHERE id = ?", (new_value, item_id))
              conn.commit()
              self.db_tree.set(item, column, new_value)
              messagebox.showinfo("Сохранено", "Значение обновлено.")
          except Exception as e:
              messagebox.showerror("Ошибка", f"Не удалось обновить значение: {e}")
          finally:
              conn.close()
          edit_window.destroy()

      # Получаем текущее значение ячейки
      current_value = self.db_tree.item(item, "values")[int(column[1:]) - 1]


      edit_window = tk.Toplevel(self.master)
      edit_window.title("Редактировать")
      edit_window.geometry("300x100")
      edit_window.resizable(False, False)

      # Поле для ввода нового значения
      edit_label = ttk.Label(edit_window, text="Новое значение:")
      edit_label.pack(pady=5)
      edit_entry = ttk.Entry(edit_window)
      edit_entry.insert(0, current_value)
      edit_entry.pack(pady=5)
      edit_entry.focus_set()

      # Кнопка сохранения
      save_button = ttk.Button(edit_window, text="Сохранить", command=save_edit)
      save_button.pack(pady=5)

      # Привязываем Enter к сохранению и Esc к закрытию окна
      edit_entry.bind("<Return>", save_edit)
      edit_window.bind("<Escape>", lambda e: edit_window.destroy())

    def sort_column(self, col, reverse):
        """Сортирует столбец."""
        data = [(self.db_tree.set(child, col), child) for child in self.db_tree.get_children('')]
        try:
            # Пытаемся сортировать как числа
            data.sort(key=lambda x: float(x[0]), reverse=reverse)
        except ValueError:
            # Если не получается, сортируем как строки
            data.sort(reverse=reverse)

        for index, (val, item) in enumerate(data):
            self.db_tree.move(item, '', index)

        # Меняем направление сортировки
        self.db_tree.heading(col, command=lambda c=col: self.sort_column(c, not reverse))

    def load_db_data(self):
        """Загружает данные из базы данных в таблицу."""
        conn = sqlite3.connect("streams.db")
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id, main_url, video_url, \"480p\", \"720p\", \"1080p\", \"1440p\", \"2160p\", last_checked, status FROM streams")
            rows = cursor.fetchall()
            for row in rows:
                self.db_tree.insert("", "end", values=row)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {e}")
        finally:
            conn.close()

    def start_scraping(self):
        urls = self.url_entry.get().split()
        max_depth = self.depth_var.get()
        concurrency = self.concurrency_var.get()

        if not urls:
            messagebox.showerror("Ошибка", "Введите URL!")
            return

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress_bar.start()

        # Очищаем лог перед новым запуском
        self.clear_log()

        # Запускаем задачу скрапинга в отдельном потоке
        self.crawler_thread = threading.Thread(
            target=self.run_crawler,
            args=(urls, max_depth, concurrency)
        )
        self.crawler_thread.daemon = True
        self.crawler_thread.start()

    def run_crawler(self, urls: List[str], max_depth: int, concurrency: int):
        """Запускает SiteCrawler в asyncio loop."""

        async def async_crawl():
            try:
                if len(urls) == 1:
                    site = urls[0]
                    crawler = SiteCrawler(
                        start_url=site,
                        max_depth=max_depth,
                        exclude_patterns=[
                            "settings", "swag", "terms", "privacy", "conditions", "support",
                            "dmca", "remove-content", "feedback", "security-center",
                            "law-enforcement", "report", "nonconsensual", "abusive-content",
                            "billing", "disable-account", "apps", "contest", "affiliates",
                            "jobs", "sitemap", "language"
                        ],
                        db_name="streams.db",
                        popup_selectors=[
                            {"selector": "button.close, button.close-button, .modal-close, .close-modal",
                             "action": "click"},
                            {"selector": "button.accept, button.agree, button#accept, button#agree", "action": "click"},
                            {"selector": "button.confirm, button#confirm", "action": "click"},
                            {"selector": "button#age-confirm, button#yes", "action": "click"},
                        ],
                        preview_image_pattern=r"https://thumb\.live\.mmcdn\.com/.*\.png",
                        content_class="content",
                        concurrency=concurrency,
                        db_updated_callback=self.update_gui_db_table  # Передаем callback

                    )
                    self.crawlers.append(crawler)
                    await crawler.crawl()
                else:
                    logger.info("[main] Параллельный кроулинг нескольких сайтов.")
                    crawlers = [
                        SiteCrawler(
                            start_url=site,
                            max_depth=max_depth,
                            exclude_patterns=[
                                "settings", "swag", "terms", "privacy", "conditions", "support",
                                "dmca", "remove-content", "feedback", "security-center",
                                "law-enforcement", "report", "nonconsensual", "abusive-content",
                                "billing", "disable-account", "apps", "contest", "affiliates",
                                "jobs", "sitemap", "language"
                            ],
                            db_name="streams.db",
                            popup_selectors=[
                                {"selector": "button.close, button.close-button, .modal-close, .close-modal",
                                 "action": "click"},
                                {"selector": "button.accept, button.agree, button#accept, button#agree",
                                 "action": "click"},
                                {"selector": "button.confirm, button#confirm", "action": "click"},
                                {"selector": "button#age-confirm, button#yes", "action": "click"},
                            ],
                            preview_image_pattern=r"https://thumb\.live\.mmcdn\.com/.*\.png",
                            content_class="content",
                            concurrency=concurrency,
                            db_updated_callback=self.update_gui_db_table

                        ) for site in urls
                    ]
                    self.crawlers.extend(crawlers)
                    await asyncio.gather(*(crawler.crawl() for crawler in crawlers))
            except Exception as e:
                logger.error(f"Ошибка в основном цикле: {e}")
            finally:
                for crawler in self.crawlers:
                    if crawler.browser:
                        await crawler.browser.close()
                        crawler.browser = None
                    if crawler.playwright:
                        await crawler.playwright.stop()
                        crawler.playwright = None
                self.crawlers = []
                self.master.after(0, self.scraping_finished)

        asyncio.run(async_crawl())

    def stop_scraping(self):
        """Останавливает процесс скрапинга."""
        for crawler in self.crawlers:
            # Очищаем очередь задач
            while not crawler.to_visit.empty():
                try:
                    crawler.to_visit.get_nowait()
                    crawler.to_visit.task_done()
                except asyncio.QueueEmpty:
                    break

        self.crawlers = []  # Очищаем список после остановки
        self.scraping_finished()

    def scraping_finished(self):
        """Действия после завершения скрапинга."""
        self.progress_bar.stop()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        messagebox.showinfo("Готово", "Скрапинг завершен!")

    def clear_log(self):
        """Очищает область лога."""
        self.log_area.configure(state='normal')
        self.log_area.delete('1.0', tk.END)
        self.log_area.configure(state='disabled')

    def save_log(self):
        """Сохраняет лог в файл."""
        file_path = filedialog.asksaveasfilename(defaultextension=".log", filetypes=[("Log Files", "*.log"), ("All Files", "*.*")])
        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(self.log_area.get("1.0", tk.END))
                messagebox.showinfo("Сохранено", f"Лог сохранен в {file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить лог: {e}")

class TextRedirector:
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, str)
        self.widget.configure(state='disabled')
        self.widget.see(tk.END) # Автопрокрутка вниз

    def flush(self):
        pass

# --- Запуск GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    gui = CrawlerGUI(root)
    root.mainloop()
