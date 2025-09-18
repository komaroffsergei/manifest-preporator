import os
import json
import random
import time
import logging
import shutil
import asyncio
from typing import Dict, List, Optional

from gtts import gTTS
from pydub import AudioSegment

# Попробуем подключить Microsoft Edge TTS (онлайн, разные голоса)
try:
    import edge_tts  # pip install edge-tts
    EDGE_TTS_AVAILABLE = True
except Exception:  # ImportError или другие проблемы
    edge_tts = None
    EDGE_TTS_AVAILABLE = False

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Пути
DATASET_DIR = "result/dataset"
OUTPUT_DIR = os.path.join(DATASET_DIR, "audio")
MANIFEST_FILE = os.path.join(DATASET_DIR, "manifest.csv")
WORDS_JSON = "words.json"  # JSON: {"инфинитив": ["предложение1", "предложение2", ...]}


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_dataset() -> None:
    # Полная очистка каталога dataset
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR, ignore_errors=True)
    ensure_dirs()


# Профили голосов
# gTTS — один русский голос, но мы различаем профили через скорость речи
VOICE_PROFILES = [
    {"name": "gtts_fast", "engine": "gtts", "params": {"slow": False}},
    {"name": "gtts_slow", "engine": "gtts", "params": {"slow": True}},
]

# Если доступен edge-tts, добавим настоящие разные русские голоса
if EDGE_TTS_AVAILABLE:
    VOICE_PROFILES.extend([
        {"name": "ms_dmitry", "engine": "edge_tts", "params": {"voice": "ru-RU-DmitryNeural"}},
        {"name": "ms_svetlana", "engine": "edge_tts", "params": {"voice": "ru-RU-SvetlanaNeural"}},
    ])


# Озвучивание текста через gTTS и сохранение в .wav (16kHz mono)
def tts_gtts(text: str, filename_no_ext: str, slow: bool = False) -> Optional[str]:
    try:
        tts = gTTS(text=text, lang='ru', slow=slow)
        temp_mp3 = f"{filename_no_ext}.mp3"
        tts.save(temp_mp3)

        audio = AudioSegment.from_mp3(temp_mp3)
        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_filename = f"{filename_no_ext}.wav"
        audio.export(wav_filename, format="wav")

        os.remove(temp_mp3)
        logger.info(f"✅ Сохранено: {wav_filename}")
        return wav_filename
    except Exception as e:
        logger.error(f"❌ Ошибка gTTS при озвучивании '{text}': {e}")
        return None


async def _tts_edge_async(text: str, mp3_path: str, voice: str) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(mp3_path)


def tts_edge(text: str, filename_no_ext: str, voice: str) -> Optional[str]:
    if not EDGE_TTS_AVAILABLE:
        logger.warning("edge-tts недоступен, падение на gTTS")
        return tts_gtts(text, filename_no_ext, slow=False)
    try:
        temp_mp3 = f"{filename_no_ext}.mp3"
        asyncio.run(_tts_edge_async(text, temp_mp3, voice))

        audio = AudioSegment.from_mp3(temp_mp3)
        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_filename = f"{filename_no_ext}.wav"
        audio.export(wav_filename, format="wav")

        os.remove(temp_mp3)
        logger.info(f"✅ Сохранено: {wav_filename}")
        return wav_filename
    except Exception as e:
        logger.error(f"❌ Ошибка edge-tts при озвучивании '{text}': {e}")
        return None


def text_to_speech(text: str, filename_no_ext: str, voice_profile: dict) -> Optional[str]:
    engine = voice_profile.get("engine", "gtts")
    params = voice_profile.get("params", {})
    if engine == "edge_tts":
        return tts_edge(text, filename_no_ext, voice=params.get("voice", "ru-RU-DmitryNeural"))
    # gTTS по умолчанию
    return tts_gtts(text, filename_no_ext, slow=bool(params.get("slow", False)))


def load_words_json(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл {path} не найден. Создайте JSON вида: {{\"инфинитив\": [\"предложение1\", \"предложение2\"]}}"
        )
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("words.json должен быть объектом {слово: [список_предложений]}")
        # нормализуем значения
        normalized: Dict[str, List[str]] = {}
        for word, sentences in data.items():
            if not isinstance(word, str):
                logger.warning(f"Пропуск некорректного ключа (ожидалась строка): {word}")
                continue
            if not isinstance(sentences, list):
                logger.warning(f"Пропуск '{word}': значения должны быть списком предложений")
                continue
            # фильтруем пустые строки, нормализуем пробелы
            clean_sentences = []
            for s in sentences:
                if isinstance(s, str):
                    s2 = s.strip()
                    if s2:
                        clean_sentences.append(s2)
            if not clean_sentences:
                logger.warning(f"Для слова '{word}' нет валидных предложений — пропуск")
                continue
            normalized[word.strip()] = clean_sentences
        return normalized


# Главная функция
def main():
    # 1) Очистка dataset перед генерацией
    clean_dataset()

    # 2) Загрузка слов и предложений
    try:
        words_to_sentences = load_words_json(WORDS_JSON)
    except Exception as e:
        logger.error(str(e))
        return

    if not words_to_sentences:
        logger.warning("В words.json нет валидных данных для озвучивания.")
        return

    logger.info(f"Загружено {len(words_to_sentences)} слов из {WORDS_JSON}.")

    manifest_lines: List[str] = []
    total_files = 0

    for word, sentences in words_to_sentences.items():
        logger.info(f"🔄 Обработка слова: {word}")

        for i, sentence in enumerate(sentences, start=1):
            profile = random.choice(VOICE_PROFILES)
            profile_name = profile.get("name", "voice")
            safe_word = word.replace('/', '_').replace('\\', '_')
            filename_no_ext = os.path.join(OUTPUT_DIR, f"{safe_word}_{i}_{profile_name}")

            wav_file = text_to_speech(sentence, filename_no_ext, profile)

            if wav_file:
                # Путь в манифесте должен быть относительным относительно manifest.csv
                manifest_dir = os.path.dirname(MANIFEST_FILE)
                relative_path = os.path.relpath(wav_file, start=manifest_dir).replace("\\", "/")
                manifest_lines.append(f"{relative_path},{sentence}")
                total_files += 1

            # Небольшая пауза, чтобы не заддосить сервисы TTS
            time.sleep(0.2)

    # Запись manifest.csv
    with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
        f.write("audio_path,text\n")  # заголовок
        for line in manifest_lines:
            f.write(line + "\n")

    logger.info(f"🎉 Готово! Создано {total_files} аудиофайлов.")
    logger.info(f"📄 manifest.csv сохранён с {len(manifest_lines)} записями.")


if __name__ == "__main__":
    main()
