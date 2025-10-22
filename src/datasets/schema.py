from dataclasses import dataclass
from typing import Optional, Dict

@dataclass(frozen=True)
class Document:
    doc_id: str
    title: Optional[str]
    text: str
    metadata: Optional[Dict] = None

@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
