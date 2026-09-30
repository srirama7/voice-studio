"""Text Processing Module for Voice Studio.

Provides exact verbatim processing, Ollama LLM narrative script cleaning with fallback,
and Critical Token Protection regex gate checking for numbers, dates, currency,
percentages, and proper names/acronyms.
"""

from __future__ import annotations

import enum
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

logger = logging.getLogger(__name__)


class TextProcessingMode(str, enum.Enum):
    """Modes available for text processing."""
    EXACT = "exact"
    NARRATIVE = "narrative"


@dataclass
class CriticalTokenReport:
    """Report detailing critical token validation results."""
    passed: bool
    missing_tokens: Dict[str, List[str]] = field(default_factory=dict)
    original_counts: Dict[str, int] = field(default_factory=dict)
    processed_counts: Dict[str, int] = field(default_factory=dict)


class CriticalTokenProtection:
    """Regex gate ensuring numbers, dates, currency, percentages, and names are preserved."""

    # Regex patterns for critical token detection
    CURRENCY_PATTERN = re.compile(
        r'(?:[$€£¥]\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:USD|EUR|GBP|JPY|CAD|AUD|dollars|cents)\b)',
        re.IGNORECASE
    )
    PERCENTAGE_PATTERN = re.compile(
        r'\b\d+(?:\.\d+)?\s*(?:%|percent)',
        re.IGNORECASE
    )
    DATE_PATTERN = re.compile(
        r'\b(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{2,4}|\b\d{1,2}(?:st|nd|rd|th)?\s+of\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?)\s+\d{2,4})\b',
        re.IGNORECASE
    )
    NUMBER_PATTERN = re.compile(
        r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
    )
    # Proper names & acronyms: multi-word capitalized names or uppercase acronyms (e.g. John Smith, NASA, Dr. Watson)
    NAME_PATTERN = re.compile(
        r'\b(?:[A-Z]{2,}|\b(?:Mr|Mrs|Ms|Dr|Prof)\.\s+[A-Z][a-z]+|\b[A-Z][a-z]+\s+[A-Z][a-z]+)\b'
    )

    @classmethod
    def extract_tokens(cls, text: str) -> Dict[str, List[str]]:
        """Extract all critical token categories from text.

        Strips structural noise (stage directions, speaker tags) before token extraction
        to avoid misidentifying cleaned markup as missing critical content.

        Args:
            text: Input text string.

        Returns:
            Dictionary mapping category names to lists of extracted token strings.
        """
        # Strip stage directions and speaker labels for clean token extraction
        clean_target = re.sub(r'\[[^\]]*\]|\<[^\>]*\>', '', text)
        clean_target = re.sub(r'^\s*(?:[A-Z0-9_\s]{2,15}|Speaker\s+\d+|Host|Narrator):\s*', '', clean_target, flags=re.MULTILINE | re.IGNORECASE)

        # Extract specific patterns first to avoid double counting as generic numbers
        currency = [m.group(0).strip() for m in cls.CURRENCY_PATTERN.finditer(clean_target)]
        percentages = [m.group(0).strip() for m in cls.PERCENTAGE_PATTERN.finditer(clean_target)]
        dates = [m.group(0).strip() for m in cls.DATE_PATTERN.finditer(clean_target)]
        
        # Generic numbers (excluding those already part of currency/percentage/date)
        numbers = [m.group(0).strip() for m in cls.NUMBER_PATTERN.finditer(clean_target)]
        
        # Names & Acronyms
        names = [m.group(0).strip() for m in cls.NAME_PATTERN.finditer(clean_target)]

        return {
            "currency": currency,
            "percentages": percentages,
            "dates": dates,
            "numbers": numbers,
            "names": names,
        }

    @classmethod
    def verify(cls, original_text: str, processed_text: str) -> CriticalTokenReport:
        """Verify that critical tokens present in original_text exist in processed_text.

        Args:
            original_text: Raw input text before cleaning.
            processed_text: Output text after cleaning/processing.

        Returns:
            CriticalTokenReport indicating pass/fail status and missing tokens.
        """
        orig_tokens = cls.extract_tokens(original_text)
        proc_tokens = cls.extract_tokens(processed_text)

        proc_text_lower = processed_text.lower()
        missing_tokens: Dict[str, List[str]] = {}
        passed = True

        for category, tokens in orig_tokens.items():
            missing_in_cat: List[str] = []
            for token in tokens:
                token_clean = token.strip()
                if category == "names":
                    # Case sensitive for names / acronyms
                    if token_clean not in processed_text:
                        missing_in_cat.append(token_clean)
                else:
                    # Case insensitive for numbers, currency, dates, percentages
                    if token_clean.lower() not in proc_text_lower:
                        missing_in_cat.append(token_clean)

            if missing_in_cat:
                missing_tokens[category] = missing_in_cat
                passed = False

        orig_counts = {cat: len(toks) for cat, toks in orig_tokens.items()}
        proc_counts = {cat: len(toks) for cat, toks in proc_tokens.items()}

        return CriticalTokenReport(
            passed=passed,
            missing_tokens=missing_tokens,
            original_counts=orig_counts,
            processed_counts=proc_counts,
        )


class TextProcessor:
    """Processor for normalizing, cleaning, and gate-checking text for TTS synthesis."""

    def __init__(
        self,
        ollama_endpoint: str = "http://localhost:11434",
        ollama_model: str = "llama3",
        timeout: float = 10.0,
    ) -> None:
        """Initialize TextProcessor.

        Args:
            ollama_endpoint: Base URL for local Ollama service.
            ollama_model: Model name for narrative cleaning.
            timeout: Timeout in seconds for Ollama LLM HTTP request.
        """
        self.ollama_endpoint = ollama_endpoint.rstrip("/")
        self.ollama_model = ollama_model
        self.timeout = timeout

    def clean_exact(self, text: str) -> str:
        """Verbatim exact mode processing.

        Normalizes unicode quotes, dashes, and whitespace without altering
        any words, numbers, punctuation, or structure.

        Args:
            text: Input raw text.

        Returns:
            Verbatim normalized text.
        """
        if not text:
            return ""

        # Normalize unicode quotes and hyphens
        text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        text = text.replace("—", "-").replace("–", "-")
        # Replace non-breaking spaces and multi-spaces
        text = re.sub(r'[\r\n]+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def clean_narrative_regex(self, text: str) -> str:
        """Rule-based narrative script cleaner fallback.

        Removes markdown formatting, bracketed noise, stage directions,
        and speaker labels while leaving spoken narrative intact.

        Args:
            text: Input text.

        Returns:
            Cleaned narrative text.
        """
        if not text:
            return ""

        # Remove HTML/XML tags
        cleaned = re.sub(r'<[^>]+>', '', text)
        # Remove URLs
        cleaned = re.sub(r'https?://\S+|www\.\S+', '', cleaned)
        # Remove Markdown headers (# Header)
        cleaned = re.sub(r'^\s*#+\s*', '', cleaned, flags=re.MULTILINE)
        # Remove Markdown bold/italic (*text*, **text**, _text_, __text__)
        cleaned = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', cleaned)
        # Remove Markdown code blocks and inline code
        cleaned = re.sub(r'`{1,3}[^`]*`{1,3}', '', cleaned)
        # Remove stage directions in brackets [sighs], [pause], (chuckles)
        cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\((?:smiling|chuckles|laughs|pauses|sighs|whispering|gasping|clears throat)[^\)]*\)', '', cleaned, flags=re.IGNORECASE)
        # Remove speaker labels like "HOST:", "NARRATOR:", "Speaker 1:"
        cleaned = re.sub(r'^\s*(?:[A-Z0-9_\s]{2,15}|Speaker\s+\d+|Host|Narrator):\s*', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        # Normalize whitespace
        cleaned = self.clean_exact(cleaned)
        return cleaned

    def clean_narrative_llm(self, text: str) -> str:
        """Calls Ollama LLM to refine narrative text for spoken TTS.

        Args:
            text: Raw input text.

        Returns:
            LLM-cleaned narrative script string.

        Raises:
            RuntimeError: If Ollama service is unavailable, fails, or times out.
        """
        prompt = (
            "You are a professional voiceover script editor. "
            "Clean and refine the following text for spoken TTS narration. "
            "STRICT RULES:\n"
            "1. Remove visual markdown tags, stage cues, bracketed noise, and speaker labels.\n"
            "2. Preserve ALL numbers, dates, currency values, percentages, and proper names EXACTLY.\n"
            "3. Do NOT summarize, alter facts, or add introductory/explanatory commentary.\n"
            "4. Return ONLY the cleaned spoken text.\n\n"
            f"Input Text:\n{text}"
        )

        url = f"{self.ollama_endpoint}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    result = resp_data.get("response", "").strip()
                    if result:
                        return result
                    raise RuntimeError("Ollama returned an empty response.")
                raise RuntimeError(f"Ollama returned HTTP status {response.status}")
        except Exception as exc:
            logger.warning("Ollama LLM cleaning failed or timed out: %s", exc)
            raise RuntimeError(f"Ollama execution failed: {exc}") from exc

    def process(
        self,
        text: str,
        mode: Union[TextProcessingMode, str] = TextProcessingMode.EXACT,
        use_llm: bool = True,
    ) -> str:
        """Process text according to specified mode and protection rules.

        Args:
            text: Raw input text.
            mode: Processing mode ('exact' or 'narrative').
            use_llm: Whether to attempt Ollama LLM cleaning for narrative mode.

        Returns:
            Processed text ready for TTS synthesis.
        """
        mode_val = mode.value if isinstance(mode, enum.Enum) else str(mode).lower()

        if mode_val == TextProcessingMode.EXACT.value or mode_val == "exact":
            return self.clean_exact(text)

        if mode_val == TextProcessingMode.NARRATIVE.value or mode_val == "narrative":
            if use_llm:
                try:
                    llm_cleaned = self.clean_narrative_llm(text)
                    report = CriticalTokenProtection.verify(text, llm_cleaned)
                    if report.passed:
                        logger.info("Ollama LLM cleaning succeeded and passed Critical Token Gate.")
                        return llm_cleaned
                    
                    logger.warning(
                        "Ollama output rejected by Critical Token Protection gate. Missing tokens: %s. Falling back to regex cleaner.",
                        report.missing_tokens,
                    )
                except Exception as exc:
                    logger.info("LLM cleaning unavailable (%s). Falling back to rule-based regex cleaner.", exc)

            # Fallback to regex cleaning
            return self.clean_narrative_regex(text)

        raise ValueError(f"Unknown processing mode: {mode}")

    def chunk_text(self, text: str, max_chars: int = 250) -> List[str]:
        """Split text into sentence/clause-level chunks for TTS synthesis.

        Args:
            text: Cleaned input text.
            max_chars: Target maximum character count per chunk.

        Returns:
            List of sentence/clause chunks.
        """
        if not text:
            return []

        # Split on sentence end punctuation (. ! ?)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: List[str] = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If a single sentence exceeds max_chars, split on clauses/commas
                if len(sentence) > max_chars:
                    sub_clauses = re.split(r'(?<=[,;:])\s+', sentence)
                    sub_chunk = ""
                    for clause in sub_clauses:
                        if len(sub_chunk) + len(clause) + 1 <= max_chars:
                            sub_chunk = f"{sub_chunk} {clause}".strip()
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = clause
                    if sub_chunk:
                        current_chunk = sub_chunk
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
