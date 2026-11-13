"""文档解析（Day 8）。

把 PDF / TXT / Markdown 统一解析成 `list[LoadedPage]`。

三个必须带上的东西：**文本、来源、页码**。
没有 `source` 和 `page`，第 4 周就做不了「引用来源」——
面试官问「你怎么证明答案是文档里来的」，你就只能干瞪眼。

解析阶段的坑（面试常问）：
- 页眉页脚混进正文 → 用「重复行检测」洗掉（同一行在多页反复出现，多半是页眉）
- 表格被解析成乱码 → pypdf 对表格很弱，表格多的文档换 pdfplumber
- 换行符把一句话切断 → 用启发式合并（中文行尾没有句读符号就接上）
- 扫描版 PDF 提不出文字 → 需要 OCR，**知道就行，不用做**
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}

# 行尾是这些符号，说明这句话说完了，不该和下一行合并
_SENTENCE_END = "。！？；：.!?;:）)》」』\"'"
# 中文行如果短于这个长度且没有句读，很可能是被换行切断了
_MERGE_THRESHOLD = 40


@dataclass
class LoadedPage:
    text: str
    source: str
    page: int | None = None


# ============================================================================
# 入口
# ============================================================================

def load_file(path: str | Path) -> list[LoadedPage]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(p)
    if suffix in {".txt", ".md", ".markdown"}:
        return _load_text(p)

    raise ValueError(f"不支持的文件类型 `{suffix}`，支持：{', '.join(sorted(SUPPORTED_SUFFIXES))}")


def load_dir(directory: str | Path, recursive: bool = True) -> list[LoadedPage]:
    """加载目录下所有支持的文档。"""
    d = Path(directory)
    if not d.exists():
        logger.warning("文档目录不存在: %s", d)
        return []

    pattern = "**/*" if recursive else "*"
    pages: list[LoadedPage] = []
    for f in sorted(d.glob(pattern)):
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                pages.extend(load_file(f))
            except Exception:  # noqa: BLE001 — 单个文件坏了不该中断整批
                logger.exception("解析失败，跳过: %s", f)
    return pages


# ============================================================================
# PDF
# ============================================================================

def _load_pdf(path: Path) -> list[LoadedPage]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("未安装 pypdf。执行：pip install pypdf") from exc

    reader = PdfReader(str(path))
    raw: list[str] = []

    for i, page in enumerate(reader.pages):
        try:
            raw.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            logger.warning("第 %d 页解析失败: %s", i + 1, path.name)
            raw.append("")

    cleaned = _strip_repeated_lines(raw)

    pages: list[LoadedPage] = []
    for i, text in enumerate(cleaned):
        body = _normalize(_merge_broken_lines(text))
        if body.strip():
            pages.append(LoadedPage(text=body, source=path.name, page=i + 1))
    return pages


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """去掉页眉页脚。

    做法：统计所有页里按行出现的行，出现频率超过 60% 且行本身很短 → 判定为页眉页脚。
    """
    if len(pages) < 3:
        return pages

    counter: Counter[str] = Counter()
    for page in pages:
        for line in {ln.strip() for ln in page.splitlines() if ln.strip()}:
            counter[line] += 1

    threshold = max(2, int(len(pages) * 0.6))
    repeated = {ln for ln, c in counter.items() if c >= threshold and len(ln) <= 40}
    if not repeated:
        return pages

    logger.info("剔除疑似页眉页脚 %d 条", len(repeated))
    return [
        "\n".join(ln for ln in page.splitlines() if ln.strip() not in repeated)
        for page in pages
    ]


def _merge_broken_lines(text: str) -> str:
    """把被换行切断的句子接回去。

    规则：当前行不以句读结尾，且下一行不以标题符号/编号开头 → 合并。
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        if out and out[-1] and not out[-1].endswith(tuple(_SENTENCE_END)):
            prev = out[-1]
            if len(prev) < _MERGE_THRESHOLD or not re.match(r"^[\d#\-\*•（(]", stripped):
                out[-1] = prev + stripped
                continue
        out.append(stripped)

    return "\n".join(out)


# ============================================================================
# 纯文本
# ============================================================================

def _load_text(path: Path) -> list[LoadedPage]:
    text = path.read_text(encoding="utf-8", errors="replace")
    body = _normalize(_merge_broken_lines(text))
    if not body.strip():
        return []
    return [LoadedPage(text=body, source=path.name, page=None)]


# ============================================================================
# 通用清洗
# ============================================================================

def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
