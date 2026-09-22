from __future__ import annotations

import re

from .language_detector import Language


def _clean(value: str) -> str:
    return re.sub(r"^the\s+", "", value.strip().rstrip("."), flags=re.I)


def realize_semistructured(canonical: str, language: Language) -> str | None:
    """Render only high-confidence English relation shapes without new reasoning."""
    if language not in {"bangla", "banglish"}:
        return None
    text = " ".join(str(canonical or "").split())

    match = re.fullmatch(r"(.+?) (?:is|was) located at (.+?)[.]?", text, re.I)
    if match:
        subject, value = map(_clean, match.groups())
        return (
            f"{subject} {value}-এ অবস্থিত।"
            if language == "bangla"
            else f"{subject} {value}-e located."
        )

    match = re.fullmatch(r"(.+?) was established in (.+?)[.]?", text, re.I)
    if match:
        subject, value = map(_clean, match.groups())
        return (
            f"{subject} {value} সালে প্রতিষ্ঠিত হয়েছিল।"
            if language == "bangla"
            else f"{subject} {value}-e establish hoyechilo."
        )

    match = re.fullmatch(r"(.+?) started its operation in (.+?) under (.+?)[.]?", text, re.I)
    if match:
        subject, year, authority = map(_clean, match.groups())
        return (
            f"{subject} {authority}-এর অধীনে {year} সালে কার্যক্রম শুরু করেছিল।"
            if language == "bangla"
            else f"{subject} {authority}-er odhine {year}-e operation start korechilo."
        )

    match = re.fullmatch(
        r"(.+?) has received (the Certificate of Accreditation) from (.+?) for (.+?)[.]?",
        text,
        re.I,
    )
    if match:
        subject, credential, authority, scope = map(_clean, match.groups())
        return (
            f"{subject} {scope}-এর জন্য {authority} থেকে {credential} পেয়েছে।"
            if language == "bangla"
            else f"{subject} {scope}-er jonno {authority} theke {credential} peyechhe."
        )

    match = re.fullmatch(r"(.+?) (?:is|was) published by (.+?)[.]?", text, re.I)
    if match:
        subject, publisher = map(_clean, match.groups())
        return (
            f"{subject} {publisher} প্রকাশ করেছে।"
            if language == "bangla"
            else f"{subject}-ta {publisher} publish koreche."
        )

    match = re.fullmatch(r"(?:the\s+)?fee for (.+?) is (.+?)[.]?", text, re.I)
    if match:
        subject, value = map(_clean, match.groups())
        return (
            f"{subject}-এর ফি হলো {value}।"
            if language == "bangla"
            else f"{subject}-er fee holo {value}."
        )

    match = re.fullmatch(r"(?:the\s+)?(.+?) of (?:the\s+)?university is named (.+?)[.]?", text, re.I)
    if match:
        role, person = map(_clean, match.groups())
        return (
            f"বিশ্ববিদ্যালয়ের {role} হলেন {person}।"
            if language == "bangla"
            else f"University-er {role} hisebe {person}-er naam deya ache."
        )
    return None


__all__ = ["realize_semistructured"]
