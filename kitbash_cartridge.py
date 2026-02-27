"""
Kitbash Cartridge System
Core knowledge storage and retrieval module

Provides: 
- Content-addressed fact storage (SQLite)
- Fast keyword-based retrieval with indices
- Annotation tracking with epistemological levels
- Access logging for phantom detection
- Hot/cold fact classification
"""

import json
import sqlite3
import hashlib
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
from enum import Enum


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class EpistemicLevel(Enum):
    """Epistemological truth hierarchy per Kitbash spec"""
    L0_EMPIRICAL = 0    # Universal physical laws
    L1_NARRATIVE = 1    # World facts, history
    L2_AXIOMATIC = 2    # Behavioral rules, identity
    L3_PERSONA = 3      # Character beliefs, noise (ephemeral)


class FactStatus(Enum):
    """Fact lifecycle states"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class CartridgeState(Enum):
    """Cartridge organizational states"""
    INTACT = "intact"
    HOT = "hot"
    COLD = "cold"
    CONSOLIDATED = "consolidated"
    ARCHIVED = "archived"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FactMetadata:
    """Metadata for a single fact"""
    fact_id: int
    content_hash: str
    created_at: str
    access_count: int = 0
    last_accessed: Optional[str] = None
    status: FactStatus = FactStatus.ACTIVE
    epistemic_level: EpistemicLevel = EpistemicLevel.L2_AXIOMATIC


@dataclass
class Derivation:
    """Logical derivation relationship"""
    type: str  # positive_dependency, boundary, range_constraint, etc
    description: str
    strength: float = 1.0
    target: Optional[str] = None
    applies_to: List[str] = field(default_factory=list)
    not_applies_to: List[str] = field(default_factory=list)
    parameter: Optional[str] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    unit: Optional[str] = None


@dataclass
class Relationship:
    """Cross-fact relationship"""
    type: str  # affects, required_by, depends_on, etc
    target_fact_id: int
    description: str


@dataclass
class AnnotationMetadata:
    """Annotation data structure per spec"""
    fact_id: int
    confidence: float = 0.5
    sources: List[str] = field(default_factory=list)
    temporal_validity_start: Optional[str] = None
    temporal_validity_end: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_validated: Optional[str] = None
    epistemic_level: EpistemicLevel = EpistemicLevel.L2_AXIOMATIC
    derivations: List[Derivation] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    context_domain: str = ""
    context_subdomains: List[str] = field(default_factory=list)
    context_applies_to: List[str] = field(default_factory=list)
    context_excludes: List[str] = field(default_factory=list)
    nwp_encoding: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict"""
        return {
            "fact_id": self.fact_id,
            "metadata": {
                "confidence": self.confidence,
                "sources": self.sources,
                "temporal_validity": {
                    "start": self.temporal_validity_start,
                    "end": self.temporal_validity_end,
                } if self.temporal_validity_start else None,
                "created_at": self.created_at,
                "last_validated": self.last_validated,
                "epistemic_level": self.epistemic_level.value,
            },
            "derivations": [asdict(d) for d in self.derivations],
            "relationships": [asdict(r) for r in self.relationships],
            "context": {
                "domain": self.context_domain,
                "subdomains": self.context_subdomains,
                "applies_to": self.context_applies_to,
                "excludes": self.context_excludes,
            },
            "nwp_encoding": self.nwp_encoding,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnnotationMetadata":
        """Create from JSON dict"""
        meta = data.get("metadata", {})
        temporal = meta.get("temporal_validity") or {}
        context = data.get("context", {})
        
        derivations = [
            Derivation(
                type=d.get("type", ""),
                description=d.get("description", ""),
                strength=d.get("strength", 1.0),
                target=d.get("target"),
                applies_to=d.get("applies_to", []),
                not_applies_to=d.get("not_applies_to", []),
                parameter=d.get("parameter"),
                min_val=d.get("min"),
                max_val=d.get("max"),
                unit=d.get("unit"),
            )
            for d in data.get("derivations", [])
        ]
        
        relationships = [
            Relationship(
                type=r.get("type", ""),
                target_fact_id=r.get("target_fact_id", 0),
                description=r.get("description", ""),
            )
            for r in data.get("relationships", [])
        ]
        
        return cls(
            fact_id=data.get("fact_id", 0),
            confidence=meta.get("confidence", 0.5),
            sources=meta.get("sources", []),
            temporal_validity_start=temporal.get("start"),
            temporal_validity_end=temporal.get("end"),
            created_at=meta.get("created_at", datetime.now(timezone.utc).isoformat()),
            last_validated=meta.get("last_validated"),
            epistemic_level=EpistemicLevel(meta.get("epistemic_level", 2)),
            derivations=derivations,
            relationships=relationships,
            context_domain=context.get("domain", ""),
            context_subdomains=context.get("subdomains", []),
            context_applies_to=context.get("applies_to", []),
            context_excludes=context.get("excludes", []),
            nwp_encoding=data.get("nwp_encoding"),
        )


# ============================================================================
# ACCESS LOG TRACKING (Delta Registry)
# ============================================================================

@dataclass
class AccessLogEntry:
    """Track query patterns for phantom detection"""
    access_count: int = 0
    last_accessed: Optional[str] = None
    query_patterns: List[Dict] = field(default_factory=list)
    cycle_consistency: float = 0.0
    phantom_candidates: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AccessLogEntry":
        return cls(**data)


# ============================================================================
# CARTRIDGE CLASS
# ============================================================================

class Cartridge:
    """
    Core knowledge cartridge for Kitbash system.
    
    Manages:
    - Content-addressed fact storage via SQLite
    - Annotation metadata tracking
    - Keyword-based fact retrieval
    - Access logging for phantom detection
    - Hot/cold fact classification
    
    Design follows 2-3% overhead rule:
    - 2-3%: Indices + metadata
    - 97-98%: Facts + annotations
    """

    def __init__(self, name: str, path: str = "./cartridges"):
        """
        Initialize or open a cartridge.
        
        Args:
            name: Cartridge name (becomes directory name with .kbc extension)
            path: Parent directory for cartridges
        """
        self.name = name
        self.cartridge_dir = Path(path) / f"{name}.kbc"
        self.indices_dir = self.cartridge_dir / "indices"
        self.grains_dir = self.cartridge_dir / "grains"
        
        # File paths
        self.db_path = self.cartridge_dir / "facts.db"
        self.annotations_path = self.cartridge_dir / "annotations.jsonl"
        self.manifest_path = self.cartridge_dir / "manifest.json"
        self.metadata_path = self.cartridge_dir / "metadata.json"
        
        # Index file paths
        self.keyword_index_path = self.indices_dir / "keyword.idx"
        self.content_hash_index_path = self.indices_dir / "content_hash.idx"
        self.access_log_path = self.indices_dir / "access_log.idx"
        
        # In-memory indices (loaded from disk)
        self.keyword_index: Dict[str, Set[int]] = {}
        self.content_hash_index: Dict[str, int] = {}
        self.access_log: Dict[int, AccessLogEntry] = {}
        
        # Database connection
        self.db: Optional[sqlite3.Connection] = None
        
        # Metadata and manifest
        self.metadata: Dict = {}
        self.manifest: Dict = {}
        
        # Annotations cache (JSONL lines)
        self.annotations: Dict[int, AnnotationMetadata] = {}

    def create(self) -> None:
        """Create new cartridge directory structure and database."""
        self.cartridge_dir.mkdir(parents=True, exist_ok=True)
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.grains_dir.mkdir(parents=True, exist_ok=True)
        
        self._create_database()
        self._init_indices()
        self._init_metadata()
        self._init_manifest()
        
        print(f"âœ“ Created cartridge: {self.cartridge_dir}")

    def load(self) -> None:
        """Load existing cartridge from disk - handles fresh cartridges gracefully."""
        if not self.cartridge_dir.exists():
            raise FileNotFoundError(f"Cartridge not found: {self.cartridge_dir}")
        
        # Ensure database exists - create if missing
        if not self.db_path.exists():
            self._create_database()
        
        self._open_database()
        self._load_indices()
        
        # Load metadata with fallback - initialize if missing
        if self.metadata_path.exists():
            try:
                self._load_metadata()
            except (json.JSONDecodeError, IOError):
                self._init_metadata()
        else:
            self._init_metadata()
        
        # Load manifest with fallback - initialize if missing
        if self.manifest_path.exists():
            try:
                self._load_manifest()
            except (json.JSONDecodeError, IOError):
                self._init_manifest()
        else:
            self._init_manifest()
        
        # Load annotations with fallback - initialize if missing
        try:
            self._load_annotations()
        except (FileNotFoundError, json.JSONDecodeError, IOError):
            self.annotations = {}
        
        print(f"✓ Loaded cartridge: {self.cartridge_dir}")

    def save(self) -> None:
        """Persist all changes to disk."""
        if self.db:
            self.db.commit()
        
        self._save_indices()
        self._save_metadata()
        self._save_manifest()
        self._save_annotations()
        
        print(f"âœ“ Saved cartridge: {self.cartridge_dir}")

    # ========================================================================
    # DATABASE OPERATIONS
    # ========================================================================

    def _create_database(self) -> None:
        """Create SQLite schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                status TEXT DEFAULT 'active'
            )
        """)
        
        # Indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON facts(content_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_count ON facts(access_count DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON facts(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON facts(last_accessed DESC)")
        
        conn.commit()
        conn.close()
        self._open_database()

    def _open_database(self) -> None:
        """Open database connection."""
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of fact content."""
        return "sha256:" + hashlib.sha256(content.encode()).hexdigest()

    def add_fact(self, content: str, annotation: Optional[AnnotationMetadata] = None) -> int:
        """
        Add a new fact to cartridge. Deduplicates by content hash.
        
        Args:
            content: Fact content (string)
            annotation: Optional annotation metadata
            
        Returns:
            fact_id (int)
        """
        content_hash = self._compute_content_hash(content)
        
        # Check for duplicate
        if content_hash in self.content_hash_index:
            return self.content_hash_index[content_hash]
        
        # Insert into database
        cursor = self.db.cursor()
        cursor.execute(
            "INSERT INTO facts (content_hash, content) VALUES (?, ?)",
            (content_hash, content)
        )
        fact_id = cursor.lastrowid
        self.db.commit()
        
        # Update indices
        self.content_hash_index[content_hash] = fact_id
        
        # Extract keywords and add to index
        keywords = self._extract_keywords(content)
        for keyword in keywords:
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = set()
            self.keyword_index[keyword].add(fact_id)
        
        # Add annotation if provided
        if annotation:
            annotation.fact_id = fact_id
            self.annotations[fact_id] = annotation
        else:
            # Create default annotation
            self.annotations[fact_id] = AnnotationMetadata(fact_id=fact_id)
        
        # Initialize access log entry
        self.access_log[fact_id] = AccessLogEntry()
        
        # Add keywords from annotation context to index
        if annotation:
            for keyword in annotation.context_applies_to:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = set()
                self.keyword_index[keyword].add(fact_id)
        
        return fact_id

    def get_fact(self, fact_id: int) -> Optional[str]:
        """
        Retrieve fact content by ID.
        
        Args:
            fact_id: ID of fact to retrieve
            
        Returns:
            Fact content string, or None if not found
        """
        cursor = self.db.cursor()
        cursor.execute("SELECT content FROM facts WHERE id = ?", (fact_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_facts(self, fact_ids: List[int]) -> Dict[int, str]:
        """
        Retrieve multiple facts by ID.
        
        Args:
            fact_ids: List of fact IDs
            
        Returns:
            Dict mapping fact_id -> content
        """
        if not fact_ids:
            return {}
        
        cursor = self.db.cursor()
        placeholders = ','.join('?' * len(fact_ids))
        cursor.execute(
            f"SELECT id, content FROM facts WHERE id IN ({placeholders})",
            fact_ids
        )
        return {row[0]: row[1] for row in cursor.fetchall()}
    
    def get_all_facts(self) -> Dict[int, str]:
        """
        Retrieve all facts in cartridge.
        
        Returns:
            Dict mapping fact_id -> content
        """
        cursor = self.db.cursor()
        cursor.execute("SELECT id, content FROM facts WHERE status != 'archived'")
        return {row[0]: row[1] for row in cursor.fetchall()}
    
    @property
    def facts(self) -> Dict[int, str]:
        """
        Compatibility property: returns all facts as a dictionary.
        For use with query engines and tools expecting dict interface.
        """
        return self.get_all_facts()

    # ========================================================================
    # QUERY OPERATIONS
    # ========================================================================

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Extract keywords from text using simple word tokenization.
        Filters out stop words and single characters.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            Set of lowercase keywords
        """
        # Simple tokenization: split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        
        # Stop words (simple list)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Filter: remove stop words, keep tokens >= 2 chars
        keywords = {t for t in tokens if len(t) >= 2 and t not in stop_words}
        return keywords

    def query(self, query_text: str, log_access: bool = True) -> List[int]:
        """
        Query cartridge for facts matching query text.
        Uses keyword-based intersection.
        
        Args:
            query_text: Natural language query
            log_access: Whether to log this query (for phantom tracking)
            
        Returns:
            List of matching fact IDs (sorted by relevance)
        """
        # Extract keywords from query
        query_keywords = self._extract_keywords(query_text)
        
        if not query_keywords:
            return []
        
        # Find facts matching all keywords (intersection)
        candidates = None
        for keyword in query_keywords:
            if keyword in self.keyword_index:
                if candidates is None:
                    candidates = self.keyword_index[keyword].copy()
                else:
                    candidates &= self.keyword_index[keyword]
        
        if not candidates:
            # Fallback: facts matching any keyword
            candidates = set()
            for keyword in query_keywords:
                if keyword in self.keyword_index:
                    candidates |= self.keyword_index[keyword]
        
        result_ids = sorted(list(candidates))
        
        # Log access pattern for phantom detection
        if log_access and result_ids:
            for fact_id in result_ids:
                self._log_access(fact_id, list(query_keywords))
        
        return result_ids

    def query_detailed(self, query_text: str) -> Dict[int, Dict]:
        """
        Query with full fact data and annotations.
        
        Args:
            query_text: Natural language query
            
        Returns:
            Dict mapping fact_id -> {"content": str, "annotation": AnnotationMetadata}
        """
        fact_ids = self.query(query_text, log_access=True)
        
        result = {}
        for fact_id in fact_ids:
            content = self.get_fact(fact_id)
            annotation = self.annotations.get(fact_id)
            result[fact_id] = {
                "content": content,
                "annotation": annotation,
            }
        
        return result

    # ========================================================================
    # ACCESS LOGGING & PHANTOM TRACKING
    # ========================================================================

    def _log_access(self, fact_id: int, query_concepts: List[str]) -> None:
        """
        Log access to a fact for phantom tracking (Delta Registry).
        
        Args:
            fact_id: Fact ID being accessed
            query_concepts: Keywords from query that matched
        """
        if fact_id not in self.access_log:
            self.access_log[fact_id] = AccessLogEntry()
        
        log = self.access_log[fact_id]
        log.access_count += 1
        log.last_accessed = datetime.now(timezone.utc).isoformat()
        
        # Update database
        cursor = self.db.cursor()
        cursor.execute(
            "UPDATE facts SET access_count = ?, last_accessed = ? WHERE id = ?",
            (log.access_count, log.last_accessed, fact_id)
        )
        self.db.commit()
        
        # Track query pattern
        pattern_key = tuple(sorted(query_concepts))
        existing = None
        for p in log.query_patterns:
            if tuple(sorted(p["concepts"])) == pattern_key:
                existing = p
                break
        
        if existing:
            existing["count"] += 1
        else:
            log.query_patterns.append({
                "concepts": query_concepts,
                "count": 1,
            })

    def get_phantom_candidates(self, min_access_count: int = 5, 
                               min_consistency: float = 0.75) -> List[Tuple[int, Dict]]:
        """
        Get facts that are phantom candidates (persistent query patterns).
        
        Args:
            min_access_count: Minimum accesses to consider
            min_consistency: Minimum consistency score
            
        Returns:
            List of (fact_id, phantom_data) tuples
        """
        candidates = []
        for fact_id, log in self.access_log.items():
            if log.access_count >= min_access_count:
                # Calculate consistency: do patterns repeat?
                if log.query_patterns:
                    total_pattern_count = sum(p["count"] for p in log.query_patterns)
                    max_pattern_count = max(p["count"] for p in log.query_patterns)
                    consistency = max_pattern_count / total_pattern_count
                else:
                    consistency = 0.0
                
                log.cycle_consistency = consistency
                
                if consistency >= min_consistency:
                    candidates.append((fact_id, {
                        "access_count": log.access_count,
                        "consistency": consistency,
                        "patterns": log.query_patterns,
                    }))
        
        # Sort by access count
        return sorted(candidates, key=lambda x: x[1]["access_count"], reverse=True)

    # ========================================================================
    # HOT/COLD CLASSIFICATION
    # ========================================================================

    def analyze_access_distribution(self) -> Dict:
        """
        Analyze fact access distribution to determine if hot/cold split needed.
        
        Returns:
            Dict with analysis results
        """
        cursor = self.db.cursor()
        cursor.execute("SELECT id, access_count FROM facts WHERE status = 'active'")
        facts = cursor.fetchall()
        
        if not facts:
            return {
                "total_facts": 0,
                "hot_ratio": 0.0,
                "should_split": False,
            }
        
        # Sort by access count
        access_counts = sorted([f[1] for f in facts], reverse=True)
        total = sum(access_counts)
        
        if total == 0:
            return {
                "total_facts": len(facts),
                "hot_ratio": 0.0,
                "should_split": False,
            }
        
        # Find cumulative distribution
        cumulative = 0
        hot_count = 0
        for count in access_counts:
            cumulative += count
            hot_count += 1
            if cumulative / total >= 0.80:  # Top 80% of accesses
                break
        
        hot_ratio = hot_count / len(facts)
        should_split = 0.15 < hot_ratio < 0.35  # Sweet spot for split
        
        return {
            "total_facts": len(facts),
            "hot_ratio": hot_ratio,
            "hot_fact_count": hot_count,
            "should_split": should_split,
            "distribution": "pareto" if hot_ratio < 0.30 else "uniform",
        }

    # ========================================================================
    # INDEX PERSISTENCE
    # ========================================================================

    def _init_indices(self) -> None:
        """Initialize empty indices."""
        self.keyword_index = {}
        self.content_hash_index = {}
        self.access_log = {}

    def _save_indices(self) -> None:
        """Save indices to disk."""
        # Keyword index
        keyword_json = {k: list(v) for k, v in self.keyword_index.items()}
        with open(self.keyword_index_path, 'w') as f:
            json.dump(keyword_json, f, indent=2)
        
        # Content hash index
        with open(self.content_hash_index_path, 'w') as f:
            json.dump(self.content_hash_index, f, indent=2)
        
        # Access log
        access_log_json = {
            str(k): v.to_dict() for k, v in self.access_log.items()
        }
        with open(self.access_log_path, 'w') as f:
            json.dump(access_log_json, f, indent=2)

    def _rebuild_content_hash_index(self) -> None:
        """Rebuild content_hash_index from database (used on startup if index file missing)."""
        if not self.db:
            return
        
        self.content_hash_index = {}
        cursor = self.db.cursor()
        cursor.execute("SELECT id, content_hash FROM facts WHERE status != 'archived'")
        
        for row in cursor.fetchall():
            fact_id = row[0]
            content_hash = row[1]
            self.content_hash_index[content_hash] = fact_id

    def _load_indices(self) -> None:
        """Load indices from disk, rebuilding from database if necessary."""
        # Keyword index
        if self.keyword_index_path.exists():
            with open(self.keyword_index_path) as f:
                data = json.load(f)
                self.keyword_index = {k: set(v) for k, v in data.items()}
        
        # Content hash index - rebuild from database if missing
        if self.content_hash_index_path.exists():
            with open(self.content_hash_index_path) as f:
                self.content_hash_index = json.load(f)
        else:
            # Rebuild from database if index file doesn't exist
            self._rebuild_content_hash_index()
        
        # Access log
        if self.access_log_path.exists():
            with open(self.access_log_path) as f:
                data = json.load(f)
                self.access_log = {
                    int(k): AccessLogEntry.from_dict(v)
                    for k, v in data.items()
                }

    # ========================================================================
    # ANNOTATION PERSISTENCE
    # ========================================================================

    def _save_annotations(self) -> None:
        """Save annotations to JSONL file."""
        with open(self.annotations_path, 'w') as f:
            for annotation in self.annotations.values():
                f.write(json.dumps(annotation.to_dict()) + '\n')

    def _load_annotations(self) -> None:
        """Load annotations from JSONL file."""
        self.annotations = {}
        if self.annotations_path.exists():
            with open(self.annotations_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        ann = AnnotationMetadata.from_dict(data)
                        self.annotations[ann.fact_id] = ann

    # ========================================================================
    # METADATA & MANIFEST
    # ========================================================================

    def _init_metadata(self) -> None:
        """Initialize metadata structure."""
        now = datetime.now(timezone.utc).isoformat()
        self.metadata = {
            "cartridge_name": self.name,
            "created_at": now,
            "last_updated": now,
            "last_accessed": now,
            "version": "1.0.0",
            "status": "active",
            "split_status": "intact",
            "health": {
                "fact_count": 0,
                "active_facts": 0,
                "archived_facts": 0,
                "annotation_count": 0,
                "grain_count": 0,
                "index_size_kb": 0,
                "total_size_mb": 0.0,
                "access_distribution": "uniform",
                "hot_fact_ratio": 0.0,
                "avg_confidence": 0.5,
                "last_validation": now,
                "validation_pass_rate": 1.0,
            },
            "performance": {
                "avg_query_latency_ms": 0,
                "p95_query_latency_ms": 0,
                "p99_query_latency_ms": 0,
                "cache_hit_rate": 0.0,
                "last_24h_queries": 0,
            },
            "flags": {
                "needs_split": False,
                "needs_consolidation": False,
                "needs_reindex": False,
                "needs_grain_crystallization": False,
            }
        }

    def _save_metadata(self) -> None:
        """Update and save metadata."""
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM facts WHERE status = 'active'")
        active_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM facts WHERE status = 'archived'")
        archived_count = cursor.fetchone()[0]
        
        self.metadata["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.metadata["health"]["fact_count"] = active_count + archived_count
        self.metadata["health"]["active_facts"] = active_count
        self.metadata["health"]["archived_facts"] = archived_count
        self.metadata["health"]["annotation_count"] = len(self.annotations)
        
        # Get index sizes
        index_size = sum(
            Path(f).stat().st_size
            for f in [self.keyword_index_path, self.content_hash_index_path, self.access_log_path]
            if f.exists()
        ) // 1024
        self.metadata["health"]["index_size_kb"] = index_size
        
        # Total size
        total_size = self.cartridge_dir.stat().st_size / (1024 * 1024)
        self.metadata["health"]["total_size_mb"] = round(total_size, 2)
        
        # Avg confidence
        if self.annotations:
            avg_conf = sum(a.confidence for a in self.annotations.values()) / len(self.annotations)
            self.metadata["health"]["avg_confidence"] = round(avg_conf, 2)
        
        # Access distribution
        dist = self.analyze_access_distribution()
        self.metadata["health"]["access_distribution"] = dist.get("distribution", "uniform")
        self.metadata["health"]["hot_fact_ratio"] = round(dist.get("hot_ratio", 0.0), 2)
        self.metadata["flags"]["needs_split"] = dist.get("should_split", False)
        
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self._init_metadata()

    def _init_manifest(self) -> None:
        """Initialize manifest structure."""
        now = datetime.now(timezone.utc).isoformat()
        self.manifest = {
            "cartridge_name": self.name,
            "version": "1.0.0",
            "api_version": "1.0",
            "created": now,
            "last_updated": now,
            "author": "Kitbash System",
            "description": f"Cartridge for {self.name}",
            "domains": [],
            "tags": [],
            "dependencies": [],
            "provides": [],
            "grain_inventory": {},
            "axiom_coverage": [],
            "license": "CC-BY-4.0",
            "compression": {
                "format": "ternary_grains",
                "ratio": 0.0,
                "original_size_mb": 0.0,
                "compressed_size_mb": 0.0,
            }
        }

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        self.manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def _load_manifest(self) -> None:
        """Load manifest from disk."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self._init_manifest()

    # ========================================================================
    # CONTEXT & STATS
    # ========================================================================

    def get_stats(self) -> Dict:
        """Get current cartridge statistics."""
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM facts WHERE status = 'active'")
        active_facts = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(access_count) FROM facts WHERE status = 'active'")
        total_accesses = cursor.fetchone()[0] or 0
        
        return {
            "name": self.name,
            "active_facts": active_facts,
            "annotations": len(self.annotations),
            "keywords": len(self.keyword_index),
            "total_accesses": total_accesses,
            "access_distribution": self.analyze_access_distribution(),
            "phantom_candidates": len(self.get_phantom_candidates()),
            "size_mb": self.metadata.get("health", {}).get("total_size_mb", 0),
        }

    def close(self) -> None:
        """Close database connection."""
        if self.db:
            self.db.close()
            self.db = None


# ============================================================================
# HELPER: Context Assembly (for LLM prompts)
# ============================================================================

def assemble_context(cartridge: Cartridge, query_text: str, 
                     token_budget: int = 3300, include_axioms: bool = True) -> Dict:
    """
    Assemble context for LLM from cartridge query.
    Respects token budgets per Kitbash tier 1-2 spec.
    
    Args:
        cartridge: Cartridge to query
        query_text: User query
        token_budget: Available tokens for context (default 3300)
        include_axioms: Whether to include axioms
        
    Returns:
        Dict with context components
    """
    # Query cartridge
    result = cartridge.query_detailed(query_text)
    
    context = {
        "query": query_text,
        "facts": [],
        "annotations": [],
        "token_count": len(query_text.split()),  # Rough estimate
    }
    
    # Add facts until token budget
    for fact_id, data in result.items():
        fact_text = data["content"]
        fact_tokens = len(fact_text.split())
        
        if context["token_count"] + fact_tokens < token_budget:
            context["facts"].append({
                "id": fact_id,
                "content": fact_text,
                "confidence": data["annotation"].confidence,
            })
            context["token_count"] += fact_tokens
        else:
            break
    
    return context


if __name__ == "__main__":
    # Example usage
    print("Kitbash Cartridge System - Example Usage\n")
    
    # Create a test cartridge
    cart = Cartridge("test_materials")
    cart.create()
    
    # Add some facts
    fact1 = cart.add_fact(
        "PLA requires 60Â°C Â±5Â°C for optimal gelling",
        AnnotationMetadata(
            fact_id=0,  # Will be set by add_fact
            confidence=0.92,
            sources=["Handbook_2023", "Research_2024"],
            context_domain="bioplastics",
            context_applies_to=["PLA", "synthetic_polymers"],
        )
    )
    
    fact2 = cart.add_fact(
        "Temperature affects polymer crystallinity",
        AnnotationMetadata(
            fact_id=0,
            confidence=0.85,
            context_domain="bioplastics",
            context_applies_to=["polymers"],
        )
    )
    
    fact3 = cart.add_fact(
        "Synthetic polymers are more stable than natural ones",
        AnnotationMetadata(
            fact_id=0,
            confidence=0.78,
            context_domain="materials",
            context_applies_to=["synthetic_polymers"],
        )
    )
    
    print(f"\nAdded 3 facts: {fact1}, {fact2}, {fact3}\n")
    
    # Query the cartridge
    query = "temperature PLA gelling"
    results = cart.query(query)
    print(f"Query: '{query}'")
    print(f"Matching facts: {results}\n")
    
    # Get detailed results
    detailed = cart.query_detailed(query)
    for fact_id, data in detailed.items():
        print(f"Fact {fact_id}:")
        print(f"  Content: {data['content']}")
        print(f"  Confidence: {data['annotation'].confidence}\n")
    
    # Check phantom candidates
    print("Phantom candidates (after access logging):")
    for fact_id, phantom_data in cart.get_phantom_candidates(min_access_count=1):
        print(f"  Fact {fact_id}: {phantom_data}\n")
    
    # Get statistics
    stats = cart.get_stats()
    print("Cartridge Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save
    cart.save()
    
    # Reload test
    print("\n--- Testing reload ---\n")
    cart2 = Cartridge("test_materials")
    cart2.load()
    
    print(f"Reloaded cartridge with {cart2.get_stats()['active_facts']} facts")
    results2 = cart2.query("synthetic polymers")
    print(f"Query results: {results2}")
    
    cart.close()
    cart2.close()
