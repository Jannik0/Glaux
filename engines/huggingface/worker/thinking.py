"""Parse/strip ``<think>``-style reasoning markup (mirrors the JS ``stripThinking`` helper).

Also strips non-thinking special-token markup from Hugging Face generation so
saved messages match what the renderer showed live, while keeping thinking
delimiters the renderer uses to split thoughts from the answer.
"""

import re

_MARKUP_TAG_RE = re.compile(r"(?:<[^>]+>|\[[^\]]+\])")


def _thinking_open_tag_from_generation_suffix(suffix: str) -> str | None:
    """Opening think tag left unclosed by the chat-template generation prompt.

    The suffix almost always begins with a role marker (e.g. ``<|im_start|>``)
    that ``skip_prompt`` omits. An odd number of further tags means the suffix
    ends with an opening marker whose body arrives in the stream, not in the
    suffix. Return that marker so it can be prepended to saved output.
    """
    if not isinstance(suffix, str) or not suffix:
        return None
    matches = list(_MARKUP_TAG_RE.finditer(suffix))
    if len(matches) <= 1:
        return None
    if (len(matches) - 1) % 2 != 1:
        return None
    tag = matches[-1].group(0)
    if len(tag) > 1 and tag[1] == "/":
        return None
    return tag


def _tag_close_char(open_char: str) -> str:
    return ">" if open_char == "<" else "]"


def _is_stop_marker_tag_inner(inner: str) -> bool:
    return inner.strip().upper() == "STOP"


def _find_markup_tag_at(text: str, from_index: int) -> tuple[int, int] | tuple[str, int] | None:
    angle_start = text.find("<", from_index)
    square_start = text.find("[", from_index)
    if angle_start == -1:
        tag_start = square_start
    elif square_start == -1:
        tag_start = angle_start
    else:
        tag_start = min(angle_start, square_start)
    if tag_start == -1:
        return None
    close_char = _tag_close_char(text[tag_start])
    tag_end = text.find(close_char, tag_start + 1)
    if tag_end == -1:
        return ("incomplete", tag_start)
    if text[tag_start] == "[" and _is_stop_marker_tag_inner(text[tag_start + 1 : tag_end]):
        return _find_markup_tag_at(text, tag_start + 1)
    return (tag_start, tag_end)


def _is_closing_markup_tag(text: str, tag_start: int) -> bool:
    return tag_start + 1 < len(text) and text[tag_start + 1] == "/"


def _is_thinking_open_marker_tag(text: str, tag_start: int, tag_end: int) -> bool:
    if _is_closing_markup_tag(text, tag_start):
        return False
    inner = text[tag_start + 1 : tag_end].lower()
    return "think" in inner


def _has_substantive_non_tag_content_after(text: str, after_index: int) -> bool:
    i = after_index
    while i < len(text):
        if text[i] in "<[":
            tag = _find_markup_tag_at(text, i)
            if tag is None:
                return False
            if tag[0] == "incomplete":
                return False
            tag_start, tag_end = tag
            i = tag_end + 1
            continue
        if not text[i].isspace():
            return True
        i += 1
    return False


def _parse_tags_and_answer(text: str) -> tuple[str, str]:
    """Split assistant stream into thoughts vs answer (mirrors renderer parseTagsAndAnswer)."""
    mode = "answer"
    thinking_parts: list[str] = []
    answer_parts: list[str] = []
    i = 0
    while i < len(text):
        tag = _find_markup_tag_at(text, i)
        if tag is None:
            chunk = text[i:]
            if mode == "thinking":
                thinking_parts.append(chunk)
            else:
                answer_parts.append(chunk)
            break
        if tag[0] == "incomplete":
            chunk = text[i:]
            if mode == "thinking":
                thinking_parts.append(chunk)
            else:
                answer_parts.append(chunk)
            break
        tag_start, tag_end = tag
        chunk = text[i:tag_start]
        if mode == "thinking":
            thinking_parts.append(chunk)
        else:
            answer_parts.append(chunk)
        close = _is_closing_markup_tag(text, tag_start)
        thinking_open = _is_thinking_open_marker_tag(text, tag_start, tag_end)
        i = tag_end + 1
        if mode == "thinking":
            if close or not thinking_open:
                mode = "answer"
        elif not close and _has_substantive_non_tag_content_after(text, i):
            mode = "thinking"
    thinking = "".join(thinking_parts).strip()
    answer = "".join(answer_parts).strip()
    return thinking, answer


def _strip_non_thinking_markup(text: str, *, hold_unresolved: bool = False) -> str:
    """Drop chat-template / EOS markup that is not a thinking delimiter.

    Mirrors renderer ``parseTagsAndAnswer``: any opening tag followed by
    non-tag content starts thinking (the tag need not contain ``think``); a
    close tag or a non-thinking-open tag ends it. Tags that never participate
    in that split (trailing ``<|im_end|>``, stray ``</s>``, …) are removed.
    ``[STOP]`` is left as-is.

    When *hold_unresolved* is true, omit a trailing incomplete tag or an
    opening answer-mode tag with no following content (streaming: wait for
    more bytes before deciding).
    """
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    if "<" not in text and "[" not in text:
        return text

    mode = "answer"
    out: list[str] = []
    i = 0
    while i < len(text):
        tag = _find_markup_tag_at(text, i)
        if tag is None:
            out.append(text[i:])
            break
        if tag[0] == "incomplete":
            tag_start = tag[1]
            out.append(text[i:tag_start])
            if not hold_unresolved:
                out.append(text[tag_start:])
            break
        tag_start, tag_end = tag
        out.append(text[i:tag_start])
        close = _is_closing_markup_tag(text, tag_start)
        thinking_open = _is_thinking_open_marker_tag(text, tag_start, tag_end)
        tag_text = text[tag_start : tag_end + 1]
        i = tag_end + 1
        if mode == "thinking":
            out.append(tag_text)
            if close or not thinking_open:
                mode = "answer"
            continue
        if close:
            continue
        if _has_substantive_non_tag_content_after(text, i):
            out.append(tag_text)
            mode = "thinking"
            continue
        if hold_unresolved:
            break
    return "".join(out)


def _iter_stripped_non_thinking_markup(chunks, *, prefix: str = ""):
    """Yield generation chunks with non-thinking special-token markup removed.

    *prefix* is an opening thinking tag supplied by the chat template (skipped
    by ``skip_prompt``). Seeding it keeps the closer in the saved message so
    the renderer can split thoughts from the answer after reload.
    """
    text = prefix if isinstance(prefix, str) else ""
    emitted = 0
    for chunk in chunks:
        if not chunk:
            continue
        text += chunk
        cleaned = _strip_non_thinking_markup(text, hold_unresolved=True)
        out = cleaned[emitted:]
        emitted = len(cleaned)
        if out:
            yield out
    cleaned = _strip_non_thinking_markup(text, hold_unresolved=False)
    out = cleaned[emitted:]
    if out:
        yield out


def _strip_thinking_from_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    if "<" not in text and "[" not in text:
        return text
    if "think" not in text.lower():
        return text
    _, answer = _parse_tags_and_answer(text)
    return answer


def _strip_thinking_from_messages(messages: list[dict]) -> list[dict]:
    """Return a copy of messages with assistant thinking markup removed (for model input)."""
    result: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            result.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, str):
            result.append({**msg, "content": _strip_thinking_from_text(content)})
            continue
        if not isinstance(content, list):
            result.append(msg)
            continue
        next_content = []
        changed = False
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                answer = _strip_thinking_from_text(part["text"])
                if answer != part["text"]:
                    changed = True
                    next_content.append({**part, "text": answer})
                else:
                    next_content.append(part)
            else:
                next_content.append(part)
        result.append({**msg, "content": next_content} if changed else msg)
    return result
