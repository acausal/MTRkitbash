# Cartridge Format & Usage Guide

## Overview

A **cartridge** is Kitbash's unit of knowledge storage. Each cartridge is a directory containing facts, annotations, metadata, and indices. Cartridges are designed for:

- **Content-addressed storage** (SHA256 deduplication, no duplicates possible)
- **Rich metadata per fact** (confidence, sources, temporal validity, rules, edges)
- **Fast keyword search** (in-memory indices)
- **Query pattern tracking** (for sleep pipeline phantom detection)

Cartridges are named `{domain}_{faction}.kbc` (e.g., `physics_general.kbc`, `biology_specialist.kbc`).

---

## Physical Structure

Each cartridge is a directory with the following layout:

```
physics_general.kbc/
├── facts.db                     # SQLite database (facts + metadata)
├── annotations.jsonl            # Line-delimited JSON (per-fact annotations)
├── manifest.json                # Cartridge provenance & versioning
├── metadata.json                # Health & performance metrics
├── indices/
│   ├── keyword.idx             # Keyword→fact_id mapping (JSON)
│   ├── content_hash.idx        # SHA256→fact_id mapping (JSON)
│   └── access_log.idx          # Query pattern tracking (JSON)
└── grains/                      # Reserved for crystallized grains (Phase 5)
```

---

## Core Files Explained

### 1. facts.db (SQLite)

**Purpose:** Single source of truth for fact content.

**Schema:**
```sql
CREATE TABLE facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT UNIQUE NOT NULL,     -- SHA256 of content
    content TEXT NOT NULL,                 -- Fact text
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,        -- Query hit count
    last_accessed TEXT,                    -- ISO 8601 timestamp
    status TEXT DEFAULT 'active'           -- 'active', 'archived', 'deprecated'
)
```

**Key properties:**
- Content is immutable once stored (facts table is append-only)
- `content_hash` is unique; adding duplicate content returns existing fact_id
- `access_count` incremented on each query hit (for hot/cold analysis)
- `status` allows soft deletes (archived facts not returned in queries)

**Size:** ~10-50 KB per 100 facts (varies with fact length).

---

### 2. annotations.jsonl (Line-Delimited JSON)

**Purpose:** Rich metadata per fact. One JSON object per line, one fact per line max.

**Example line:**
```json
{"fact_id": 5, "metadata": {"confidence": 0.92, "sources": ["neuroscience_journal_2023", "domain_expert"], "temporal_validity": {"start": "2024-01-01", "end": null}, "created_at": "2024-01-15T10:30:00Z", "last_validated": "2026-05-08T21:30:00Z", "epistemic_level": 2}, "derivations": [{"type": "positive_dependency", "description": "Requires fact 12 to be true", "strength": 0.95, "target": "fact_12", "applies_to": ["neuroscience"], "not_applies_to": ["behavioral_psychology"]}], "relationships": [{"type": "affects", "target_fact_id": 18, "description": "Increases likelihood of fact 18"}], "context": {"domain": "neuroscience", "subdomains": ["sleep", "memory"], "applies_to": ["medicine", "psychology"], "excludes": ["physics"]}, "nwp_encoding": null}
```

**Formatted for readability:**
```json
{
  "fact_id": 5,
  "metadata": {
    "confidence": 0.92,
    "sources": ["neuroscience_journal_2023", "domain_expert"],
    "temporal_validity": {
      "start": "2024-01-01",
      "end": null
    },
    "created_at": "2024-01-15T10:30:00Z",
    "last_validated": "2026-05-08T21:30:00Z",
    "epistemic_level": 2
  },
  "derivations": [
    {
      "type": "positive_dependency",
      "description": "Requires fact 12 to be true",
      "strength": 0.95,
      "target": "fact_12",
      "applies_to": ["neuroscience"],
      "not_applies_to": ["behavioral_psychology"],
      "parameter": null,
      "min_val": null,
      "max_val": null,
      "unit": null
    }
  ],
  "relationships": [
    {
      "type": "affects",
      "target_fact_id": 18,
      "description": "Increases likelihood of fact 18"
    }
  ],
  "context": {
    "domain": "neuroscience",
    "subdomains": ["sleep", "memory"],
    "applies_to": ["medicine", "psychology"],
    "excludes": ["physics"]
  },
  "nwp_encoding": null
}
```

**Key properties:**
- Each fact can have 0 or 1 annotation (facts don't require annotations)
- `confidence`: 0.0–1.0 (default 0.5)
- `temporal_validity`: Fact is valid from `start` to `end` (null = always valid)
- `epistemic_level`: 0=L0 (hardwired), 1=L1 (narrative), 2=L2 (axiomatic), 3=L3 (persona)
- `derivations`: Rules about this fact (optional; empty list if none)
- `relationships`: Edges to other facts (optional; empty list if none)
- `context`: Scope gates (domain, subdomains, applies_to, excludes)

**Derivation types:**
- `positive_dependency`: This fact requires another to be true
- `boundary`: Conditional scope (applies_to/not_applies_to for when rule activates)
- `range_constraint`: Parametric bounds (min_val, max_val, unit for this fact)

---

### 3. manifest.json (Provenance & Versioning)

**Purpose:** Cartridge discovery, dependency tracking, grain inventory.

**Example:**
```json
{
  "cartridge_name": "physics_general",
  "version": "1.2.3",
  "api_version": "1.0",
  "created": "2024-01-15T10:30:00Z",
  "last_updated": "2026-05-08T21:30:00Z",
  "author": "Kitbash System",
  "description": "General physics facts and principles",
  "domains": ["physics"],
  "tags": ["empirical", "deterministic", "verified"],
  "dependencies": ["mathematics_axioms"],
  "provides": ["thermodynamics_axioms", "newtonian_mechanics"],
  "grain_inventory": {
    "grain_001": {
      "fact_ids": [1, 2, 5],
      "confidence": 0.92,
      "label": "entropy_definition"
    },
    "grain_042": {
      "fact_ids": [42, 43, 51],
      "confidence": 0.87,
      "label": "newton_third_law"
    }
  },
  "axiom_coverage": [
    "law_of_conservation",
    "second_law_thermodynamics"
  ],
  "license": "CC-BY-4.0",
  "compression": {
    "format": "ternary_grains",
    "ratio": 0.0,
    "original_size_mb": 2.5,
    "compressed_size_mb": 0.0
  }
}
```

**Key properties:**
- `domains`: What subject areas this cartridge covers
- `dependencies`: Other cartridges this one relies on
- `provides`: What concepts/axioms this cartridge contributes
- `grain_inventory`: Crystallized grains (populated by sleep pipeline)
- `compression.ratio`: For phase 5+ ternary grain compression

---

### 4. metadata.json (Health & Performance)

**Purpose:** Operational monitoring. Auto-updated by `cartridge.save()`.

**Example (partial):**
```json
{
  "cartridge_name": "physics_general",
  "created_at": "2024-01-15T10:30:00Z",
  "last_updated": "2026-05-08T21:30:00Z",
  "last_accessed": "2026-05-08T21:45:00Z",
  "version": "1.0.0",
  "status": "active",
  "split_status": "intact",
  "health": {
    "fact_count": 247,
    "active_facts": 245,
    "archived_facts": 2,
    "annotation_count": 189,
    "grain_count": 18,
    "index_size_kb": 42,
    "total_size_mb": 2.8,
    "access_distribution": "pareto",
    "hot_fact_ratio": 0.22,
    "avg_confidence": 0.81,
    "last_validation": "2026-05-08T20:00:00Z",
    "validation_pass_rate": 0.98
  },
  "performance": {
    "avg_query_latency_ms": 3.2,
    "p95_query_latency_ms": 8.1,
    "p99_query_latency_ms": 14.3,
    "cache_hit_rate": 0.67,
    "last_24h_queries": 1203
  },
  "flags": {
    "needs_split": false,
    "needs_consolidation": false,
    "needs_reindex": false,
    "needs_grain_crystallization": false
  }
}
```

**Key properties:**
- `health.access_distribution`: "pareto" (20% of facts = 80% of queries) or "uniform"
- `health.hot_fact_ratio`: Percentage of facts accounting for 80% of accesses
- `flags.needs_split`: True if hot/cold split would improve performance (cartridge > 5MB and 15-35% hot ratio)

---

### 5. indices/keyword.idx (JSON)

**Purpose:** Fast keyword search via inverted index.

**Format:** Keyword → list of fact IDs

```json
{
  "sleep": [5, 12, 18, 24, 31, 42],
  "consolidation": [5, 18, 31],
  "memory": [12, 18, 24, 31, 42, 51],
  "hippocampus": [5, 24],
  "synaptic": [18, 24, 51],
  "plasticity": [24, 42]
}
```

**How it works:**
- Extract all tokens from fact content
- Filter: Remove stop words, keep tokens ≥2 chars
- Lowercase and stem (basic)
- Map each token to the fact_id that contains it

**Query:** "What is sleep consolidation?" → Extract keywords ("sleep", "consolidation") → Intersect fact_id lists → Get [5, 18, 31]

**Status:** Cache file; rebuilt from facts.db if missing on load.

---

### 6. indices/content_hash.idx (JSON)

**Purpose:** Content deduplication.

**Format:** SHA256 hash → fact_id

```json
{
  "sha256:abc123def456...": 5,
  "sha256:789ghi012jkl...": 12,
  "sha256:345mno678pqr...": 18
}
```

**Usage:** Before adding a new fact, compute its SHA256 hash and check if it already exists in the cartridge.

**Status:** Cache file; rebuilt from facts.db if missing on load.

---

### 7. indices/access_log.idx (JSON)

**Purpose:** Phantom detection. Tracks which query patterns consistently retrieve each fact.

**Example:**
```json
{
  "5": {
    "access_count": 47,
    "last_accessed": "2026-05-08T21:43:00Z",
    "query_patterns": [
      {"concepts": ["sleep", "memory"], "count": 18},
      {"concepts": ["sleep", "consolidation"], "count": 14},
      {"concepts": ["sleep"], "count": 15}
    ],
    "cycle_consistency": 0.38,
    "phantom_candidates": ["phantom_sleep_memory_001"]
  },
  "12": {
    "access_count": 23,
    "last_accessed": "2026-05-08T20:15:00Z",
    "query_patterns": [
      {"concepts": ["memory", "consolidation"], "count": 12},
      {"concepts": ["memory", "hippocampus"], "count": 11}
    ],
    "cycle_consistency": 0.52,
    "phantom_candidates": []
  }
}
```

**Key properties:**
- `query_patterns`: Tracks *which keywords together* retrieved this fact, and how often
- `cycle_consistency`: (max_pattern_count / total_patterns) — signals if one pattern dominates
- `phantom_candidates`: Flag if this fact is a candidate for phantom emergence

**Status:** Cache file; rebuilt from query history if missing.

---

## Data Types

### AnnotationMetadata

Complete structure (all fields shown, but most are optional):

```python
{
  "fact_id": int,                    # Required; fact ID this annotates
  "metadata": {
    "confidence": float,             # 0.0-1.0 (default 0.5)
    "sources": [str],                # List of sources (default [])
    "temporal_validity": {
      "start": str | null,           # ISO 8601 (null = always valid)
      "end": str | null
    } | null,
    "created_at": str,               # ISO 8601 (default now)
    "last_validated": str | null,    # ISO 8601 (default null)
    "epistemic_level": int           # 0-3 (default 2)
  },
  "derivations": [                   # Optional; default []
    {
      "type": str,                   # "positive_dependency", "boundary", "range_constraint"
      "description": str,
      "strength": float,             # 0.0-1.0 (default 1.0)
      "target": str | null,          # Target fact ID or label
      "applies_to": [str],           # Domains where rule applies
      "not_applies_to": [str],       # Domains where rule doesn't apply
      "parameter": str | null,       # For parametric rules
      "min_val": float | null,       # Min bound
      "max_val": float | null,       # Max bound
      "unit": str | null             # Unit (e.g., "celsius")
    }
  ],
  "relationships": [                 # Optional; default []
    {
      "type": str,                   # "affects", "required_by", "depends_on"
      "target_fact_id": int,         # Fact this relates to
      "description": str
    }
  ],
  "context": {                       # Optional; default empty
    "domain": str,                   # Primary domain (e.g., "physics")
    "subdomains": [str],             # Finer categories
    "applies_to": [str],             # Domains where this fact applies
    "excludes": [str]                # Domains where this fact doesn't apply
  },
  "nwp_encoding": str | null        # Reserved for future use
}
```

---

## Building a Cartridge

### Method 1: Programmatic (Python)

```python
from kitbash_cartridge import Cartridge, AnnotationMetadata, Derivation, Relationship, EpistemicLevel

# Initialize (creates directory if doesn't exist)
cart = Cartridge("physics", "./cartridges")
cart.create()  # Creates directory structure + empty SQLite

# Add fact #1 with annotation
fact1_id = cart.add_fact(
    content="Sleep consolidates memories through slow-wave activity",
    annotation=AnnotationMetadata(
        fact_id=0,  # Auto-assigned by SQLite
        confidence=0.92,
        sources=["neuroscience_journal_2023"],
        epistemic_level=EpistemicLevel.L2_AXIOMATIC,
        derivations=[
            Derivation(
                type="positive_dependency",
                description="Depends on intact hippocampus",
                strength=0.98,
                target="fact_hippocampus"
            )
        ],
        context={
            "domain": "neuroscience",
            "subdomains": ["sleep", "memory"],
            "applies_to": ["medicine", "psychology"]
        }
    )
)
print(f"Added fact: {fact1_id}")

# Add fact #2 without annotation (minimal)
fact2_id = cart.add_fact(
    content="The hippocampus is crucial for memory formation"
)
print(f"Added fact: {fact2_id}")

# Add fact #3 with relationship to fact #1
fact3_id = cart.add_fact(
    content="Slow-wave sleep is associated with memory consolidation",
    annotation=AnnotationMetadata(
        fact_id=0,
        confidence=0.88,
        sources=["walker_2009"],
        relationships=[
            Relationship(
                type="affects",
                target_fact_id=fact1_id,
                description="Supports the mechanism of sleep consolidation"
            )
        ]
    )
)

# Save all changes
cart.save()
print("✓ Cartridge saved")

# Query the cartridge
results = cart.query("What is memory consolidation?")
print(f"Query results: {results}")  # [fact1_id, fact3_id]

# Get detailed results
detailed = cart.query_detailed("memory consolidation")
for fact_id, data in detailed.items():
    print(f"Fact {fact_id}: {data['content']}")
    if data['annotation']:
        print(f"  Confidence: {data['annotation'].confidence}")
```

### Method 2: From Seed File (Bulk)

**Seed file format:** Plain text, one fact per line (Markdown, JSON, or plain text supported).

**Example: biology_seed.md**
```markdown
# Biology Seed Facts

Mitochondria are the powerhouse of the cell; they generate ATP through oxidative phosphorylation.

The double helix structure of DNA was discovered by Watson, Crick, and Franklin in 1953.

Photosynthesis converts light energy into chemical energy stored in glucose.

Cells are the basic unit of life and all organisms are composed of cells.

Ribosomes are the site of protein synthesis in both prokaryotes and eukaryotes.
```

**Batch load:**
```bash
python batch_cartridge_builder.py --seed-dir ./seeds/ --output ./cartridges/
```

This:
1. Finds all files matching `{domain}_seed.{ext}` (case-insensitive)
2. Extracts domain name (e.g., `biology_seed.md` → `biology`)
3. Creates `biology_general.kbc/`
4. Parses each line as a fact
5. Auto-generates basic annotations (confidence=0.75, no sources)
6. Saves cartridge

---

## Inspecting a Cartridge

### Load and query

```python
from kitbash_cartridge import Cartridge

cart = Cartridge("physics", "./cartridges")
cart.load()

# Get stats
stats = cart.get_stats()
print(f"Active facts: {stats['active_facts']}")
print(f"Annotations: {stats['annotations']}")
print(f"Keywords indexed: {stats['keywords']}")
print(f"Size: {stats['size_mb']} MB")

# Analyze access patterns
dist = cart.analyze_access_distribution()
print(f"Distribution: {dist['distribution']}")  # 'pareto' or 'uniform'
print(f"Hot fact ratio: {dist['hot_ratio']:.1%}")  # % of facts driving 80% of queries
print(f"Needs split: {dist['should_split']}")
```

### Find phantom candidates

```python
phantoms = cart.get_phantom_candidates(min_access_count=5, min_consistency=0.75)
for fact_id, phantom_data in phantoms:
    print(f"Fact {fact_id}: {phantom_data['consistency']:.2%} consistency")
    print(f"  Access count: {phantom_data['access_count']}")
    print(f"  Patterns: {phantom_data['patterns']}")
```

---

## Maintenance Operations

### Hot/Cold Split (When to do it)

The `metadata.json` flag `flags.needs_split` is set when:
- Cartridge > 5 MB, AND
- Hot fact ratio between 15-35% (sweet spot for splitting)

**Why split?** When 80% of queries hit ~20% of facts, keeping all facts in memory wastes space. Split into:
- `physics_hot.kbc` (frequently accessed facts)
- `physics_cold.kbc` (rarely accessed facts)

The QueryOrchestrator will query hot first, then cold if needed.

### Archive deprecated facts

```python
# Archive a fact without deleting it
cart.db.cursor().execute(
    "UPDATE facts SET status = 'archived' WHERE id = ?",
    (fact_id,)
)
cart.save()

# Archived facts don't appear in queries or indices
results = cart.query("...")  # Won't return archived facts
```

### Reindex (if indices become stale)

```python
# Rebuild indices from scratch
cart._init_indices()
cart._load_indices()  # Rebuilds from facts.db
cart.save()
```

---

## Important Design Decisions

### 1. Fact IDs are Local, Not Global

Fact ID 5 in `physics_general.kbc` is separate from fact ID 5 in `biology_general.kbc`. No cross-cartridge fact references.

**Why:** Cartridges are independent units. Dependencies are tracked at the cartridge level (manifest.json), not the fact level.

---

### 2. Annotations are Optional

Facts don't require annotations. If you `add_fact()` without annotation, the fact is created but has no metadata.

**Why:** Allows phased development. Load facts first, add metadata later as confidence improves.

---

### 3. Derivations Aren't Validated

If a derivation references a non-existent fact (e.g., `target="fact_999"` when only 50 facts exist), the cartridge doesn't complain. Sleep pipeline will catch this later.

**Why:** Loose coupling during development. Full validation happens offline in the sleep pipeline.

---

### 4. Keyword Index is a Cache

If `keyword.idx` is missing or corrupted, `cart.load()` rebuilds it from `facts.db` on the fly.

**Why:** Index is for speed, not correctness. Database is the source of truth.

---

### 5. Temporal Validity is Per-Annotation

Facts have individual validity windows. Fact A valid Jan-Dec, Fact B valid Feb-Oct. No cartridge-level expiry.

**Why:** Facts have different lifespans (seasonal data vs. permanent laws).

---

## Example: Complete 5-Fact Cartridge

**Building it:**
```python
from kitbash_cartridge import Cartridge, AnnotationMetadata, EpistemicLevel, Derivation, Relationship

cart = Cartridge("sleep_science", "./cartridges")
cart.create()

# Fact 1: Sleep definition
f1 = cart.add_fact(
    "Sleep is a state of reduced consciousness and metabolic activity",
    AnnotationMetadata(
        fact_id=0,
        confidence=0.95,
        sources=["walker_2017"],
        epistemic_level=EpistemicLevel.L1_NARRATIVE,
        context={"domain": "neuroscience", "subdomains": ["sleep"]}
    )
)

# Fact 2: Slow-wave sleep function
f2 = cart.add_fact(
    "Slow-wave sleep (deep sleep) is essential for memory consolidation",
    AnnotationMetadata(
        fact_id=0,
        confidence=0.91,
        sources=["walker_2009", "dang_2018"],
        epistemic_level=EpistemicLevel.L2_AXIOMATIC,
        derivations=[
            Derivation(
                type="positive_dependency",
                description="Requires hippocampus to function",
                strength=0.98,
                target="fact_hippocampus"
            )
        ],
        relationships=[
            Relationship(
                type="affects",
                target_fact_id=f1,
                description="Is a type of sleep"
            )
        ]
    )
)

# Fact 3: Hippocampus role (minimal annotation)
f3 = cart.add_fact(
    "The hippocampus is critical for forming new memories"
)

# Fact 4: REM sleep and learning
f4 = cart.add_fact(
    "REM sleep supports procedural learning and emotional memory consolidation",
    AnnotationMetadata(
        fact_id=0,
        confidence=0.87,
        sources=["stickgold_2013"],
        temporal_validity={
            "start": "2000-01-01",
            "end": None
        }
    )
)

# Fact 5: Sleep deprivation effects
f5 = cart.add_fact(
    "Chronic sleep deprivation impairs cognitive function and increases disease risk",
    AnnotationMetadata(
        fact_id=0,
        confidence=0.89,
        sources=["czeisler_2006", "walker_2017"],
        derivations=[
            Derivation(
                type="negative_consequence",
                description="Results in reduced consolidation",
                strength=0.95
            )
        ]
    )
)

cart.save()
print("✓ Sleep science cartridge created")

# Verify
stats = cart.get_stats()
print(f"Facts: {stats['active_facts']}, Annotations: {stats['annotations']}")
```

**Result in metadata.json:**
```json
{
  "health": {
    "fact_count": 5,
    "active_facts": 5,
    "annotation_count": 4,
    "avg_confidence": 0.912,
    "hot_fact_ratio": 0.0,
    "access_distribution": "uniform"
  }
}
```

---

## File Format Summary

| File | Format | Mutable | Purpose |
|------|--------|---------|---------|
| facts.db | SQLite | Yes (append-only) | Source of truth for fact content |
| annotations.jsonl | JSONL | Yes | Metadata per fact |
| keyword.idx | JSON | Cache | Fast keyword search |
| content_hash.idx | JSON | Cache | Deduplication |
| access_log.idx | JSON | Cache | Query pattern tracking |
| manifest.json | JSON | Yes | Versioning & grain inventory |
| metadata.json | JSON | Auto | Health & performance |

**Cache files** are rebuilt from source if missing.  
**Mutable files** are hand-editable but `cartridge.save()` will overwrite them.

---

## Troubleshooting

### "Cartridge not found" error
```python
cart = Cartridge("physics", "./cartridges")
cart.load()  # Raises FileNotFoundError
```

**Fix:** Create the cartridge first:
```python
cart.create()
```

### Indices are stale (outdated keyword search)
```python
# Rebuild indices from database
cart._init_indices()
cart._rebuild_content_hash_index()
cart._load_indices()
cart.save()
```

### Query returns no results even though fact exists

**Possible causes:**
1. Fact has status='archived' (hidden from queries)
2. Keywords don't match (check keyword.idx)
3. Fact has no annotation and no keywords were extracted

**Debug:**
```python
# Check all facts
cursor = cart.db.cursor()
cursor.execute("SELECT id, content, status FROM facts")
for row in cursor:
    print(f"Fact {row[0]} ({row[2]}): {row[1][:50]}")

# Check keyword index
print(f"Keywords: {list(cart.keyword_index.keys())}")

# Check if specific fact is indexed
if 5 in cart.keyword_index.values():
    print("Fact 5 is indexed")
else:
    print("Fact 5 is NOT indexed")
```

---

## Next Steps

1. **Create your first cartridge** from a seed file (see "Building a Cartridge" → Method 2)
2. **Load and query** it using the Cartridge API
3. **Inspect stats** to understand hot/cold distribution
4. **Add annotations** as you gain confidence in facts

Cartridges integrate seamlessly with the QueryOrchestrator. Once created, they're automatically discovered and loaded by the CartridgeLoader.
