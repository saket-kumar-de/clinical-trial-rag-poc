"""
Lightweight dataclasses mirroring the Aurora schema.
Used for type hints as records move through extraction -> storage.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Trial:
    nct_id: str
    title: str
    phase: Optional[str] = None
    condition: Optional[str] = None
    status: Optional[str] = None
    source: Optional[str] = None
    raw_json: Optional[dict] = None


@dataclass
class Intervention:
    nct_id: str
    name: str
    intervention_type: Optional[str] = None


@dataclass
class Endpoint:
    nct_id: str
    endpoint_type: str  # "primary" | "secondary"
    description: str


@dataclass
class EligibilityCriterion:
    nct_id: str
    criterion_type: str  # "inclusion" | "exclusion"
    description: str


@dataclass
class Chunk:
    nct_id: str
    source_doc: str
    section_type: str
    chunk_text: str
    embedding: Optional[list] = field(default=None)
