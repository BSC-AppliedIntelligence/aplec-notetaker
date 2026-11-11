"""Language display map: ISO 639-1 code -> (flag-icons code, English name).

Covers the world's most-spoken ~50 languages plus Catalan, keyed by the codes
Whisper emits. The middle value is a `flag-icons` SVG code (ISO 3166-1 alpha-2,
or a subdivision like `es-ct`), NOT an emoji: flag EMOJI don't render on Windows
(Segoe UI Emoji omits flag glyphs), so we serve vendored SVGs instead.

Conventions: Portuguese uses Brazil; Spanish uses
Spain; Catalan uses the actual Catalan flag (`es-ct`, the Senyera). Edit freely;
after adding a language, run `python -m aplec_web.languages` to see which
flag SVGs are missing from static/flags/4x3/. Unknown codes fall back to the
uppercase code with no flag.
"""
from __future__ import annotations

# language code -> (flag-icons code, English name)
LANGUAGES: dict[str, tuple[str, str]] = {
    "en": ("us", "English"),
    "zh": ("cn", "Chinese"),
    "yue": ("hk", "Cantonese"),
    "hi": ("in", "Hindi"),
    "es": ("es", "Spanish"),
    "fr": ("fr", "French"),
    "ar": ("sa", "Arabic"),
    "bn": ("bd", "Bengali"),
    "pt": ("br", "Portuguese"),
    "pt-br": ("br", "Brazilian Portuguese"),
    "ru": ("ru", "Russian"),
    "ur": ("pk", "Urdu"),
    "id": ("id", "Indonesian"),
    "de": ("de", "German"),
    "ja": ("jp", "Japanese"),
    "tr": ("tr", "Turkish"),
    "ta": ("in", "Tamil"),
    "te": ("in", "Telugu"),
    "mr": ("in", "Marathi"),
    "pa": ("in", "Punjabi"),
    "gu": ("in", "Gujarati"),
    "kn": ("in", "Kannada"),
    "ml": ("in", "Malayalam"),
    "vi": ("vn", "Vietnamese"),
    "ko": ("kr", "Korean"),
    "fa": ("ir", "Persian"),
    "it": ("it", "Italian"),
    "th": ("th", "Thai"),
    "pl": ("pl", "Polish"),
    "uk": ("ua", "Ukrainian"),
    "nl": ("nl", "Dutch"),
    "el": ("gr", "Greek"),
    "cs": ("cz", "Czech"),
    "sv": ("se", "Swedish"),
    "he": ("il", "Hebrew"),
    "da": ("dk", "Danish"),
    "fi": ("fi", "Finnish"),
    "no": ("no", "Norwegian"),
    "nn": ("no", "Norwegian Nynorsk"),
    "ro": ("ro", "Romanian"),
    "hu": ("hu", "Hungarian"),
    "sk": ("sk", "Slovak"),
    "bg": ("bg", "Bulgarian"),
    "hr": ("hr", "Croatian"),
    "sr": ("rs", "Serbian"),
    "ms": ("my", "Malay"),
    "tl": ("ph", "Tagalog"),
    "sw": ("tz", "Swahili"),
    "ha": ("ng", "Hausa"),
    "yo": ("ng", "Yoruba"),
    "am": ("et", "Amharic"),
    "my": ("mm", "Burmese"),
    "km": ("kh", "Khmer"),
    "ne": ("np", "Nepali"),
    "si": ("lk", "Sinhala"),
    "az": ("az", "Azerbaijani"),
    "kk": ("kz", "Kazakh"),
    "uz": ("uz", "Uzbek"),
    "ca": ("es-ct", "Catalan"),
}


def lang_flag(code: str | None) -> str:
    """flag-icons SVG code for a language (e.g. 'br', 'es-ct'); '' if unknown."""
    if not code:
        return ""
    return LANGUAGES.get(code.lower(), ("", ""))[0]


def lang_name(code: str | None) -> str:
    """English language name; falls back to the uppercase code."""
    if not code:
        return ""
    entry = LANGUAGES.get(code.lower())
    return entry[1] if entry else code.upper()


def lang_label(code: str | None) -> str:
    """Short chip label (uppercase ISO code)."""
    return (code or "").upper()


def flag_codes() -> set[str]:
    """Distinct flag-icons codes referenced (which SVGs must be vendored)."""
    return {flag for flag, _ in LANGUAGES.values()}


if __name__ == "__main__":  # `python -m aplec_web.languages` — audit vendored SVGs
    from pathlib import Path

    flags_dir = Path(__file__).resolve().parent / "static" / "flags" / "4x3"
    present = {p.stem for p in flags_dir.glob("*.svg")} if flags_dir.is_dir() else set()
    missing = sorted(flag_codes() - present)
    print(f"{len(flag_codes())} flag codes referenced, {len(present)} SVGs present")
    print("MISSING:", ", ".join(missing) if missing else "none ✓")
