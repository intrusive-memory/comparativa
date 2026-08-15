"""Fountain parsing: the ordered element stream every later stage consumes."""

from .elements import Element, ElementStream, ElementType, InlineNote
from .fountain import (
    PARSER_NAME,
    PARSER_VERSION,
    FountainParseError,
    parse_file,
    parse_text,
)

__all__ = [
    "PARSER_NAME",
    "PARSER_VERSION",
    "Element",
    "ElementStream",
    "ElementType",
    "FountainParseError",
    "InlineNote",
    "parse_file",
    "parse_text",
]
