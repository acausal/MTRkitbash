# Setup: Kitbash Installation & Configuration

## Prerequisites

- **Python 3.14+** (3.14.0 or later)
- **Redis 4.5+** (running locally or accessible via network)
- **~500MB disk space** minimum for initial grain cache and cartridge storage
- **Git** (to clone repository)

Verify your Python version:
```bash
python3 --version
```

Verify Redis is installed and running:
```bash
redis-cli ping
# Should respond: PONG
```

If Redis is not installed:
- **macOS (Homebrew):** `brew install redis && brew services start redis`
- **Ubuntu/Debian:** `sudo apt-get install redis-server && sudo systemctl start redis-server`
- **Windows:** Use WSL + above, or run Redis in Docker: `docker run -d -p 6379:6379 redis:latest`

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url> kitbash
cd kitbash
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `redis` (Python Redis client, ≥4.5.0)
- `torch` (PyTorch for MTR v5.5 neural components, ≥2.0.0)

### 3. Verify dependencies

```bash
python3 -c "import redis; import torch; print('Dependencies OK')"
```

If this succeeds, you're ready to configure.

---

## Configuration

Kitbash uses environment variables for runtime configuration. Create a `.env` file in the project root:

```bash
# Redis connection
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0

# Cartridge storage
export CARTRIDGE_DIR=./cartridges
export GRAIN_CACHE_SIZE=10000

# Logging
export LOG_LEVEL=INFO
export LOG_FILE=./kitbash.log

# Sleep cycle (in seconds; default 3600 = 1 hour)
export SLEEP_CYCLE_INTERVAL=3600

# Optional: BitNet (Phase 5; leave blank if not in use)
export BITNET_MODEL_PATH=
```

Load the environment:
```bash
source .env
```

### Configuration defaults (if not set)

| Variable | Default | Purpose |
|----------|---------|---------|
| `REDIS_HOST` | `localhost` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_DB` | `0` | Redis database number |
| `CARTRIDGE_DIR` | `./cartridges` | Where to store `.kbc` cartridge files |
| `GRAIN_CACHE_SIZE` | `10000` | Max grains held in memory before LRU eviction |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FILE` | `./kitbash.log` | Where to write event logs |
| `SLEEP_CYCLE_INTERVAL` | `3600` | Sleep pipeline execution interval (seconds) |

---

## First Query: Walkthrough

### 1. Create a minimal cartridge (optional)

If you have cartridges ready, place `.kbc` files in `$CARTRIDGE_DIR`. Otherwise, skip to step 2.

### 2. Start the orchestrator (in Python REPL or script)

```python
from query_orchestrator_posix import QueryOrchestrator
import os

# Initialize
orchestrator = QueryOrchestrator(
    redis_host=os.getenv("REDIS_HOST", "localhost"),
    redis_port=int(os.getenv("REDIS_PORT", 6379)),
    cartridge_dir=os.getenv("CARTRIDGE_DIR", "./cartridges"),
)

# Execute a query
result = orchestrator.execute_query("What are the effects of sleep on memory?")

print(f"Answer: {result.response}")
print(f"Confidence: {result.confidence}")
print(f"Sources: {result.source_grains}")
```

### 3. Expected output

```
Answer: Sleep consolidates episodic and semantic memory through slow-wave activity...
Confidence: 0.82
Sources: ['grain_sleep_consolidation_001', 'grain_neural_plasticity_042']
```

---

## Troubleshooting

### "Redis connection refused"

**Symptom:** `redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379`

**Fix:**
1. Verify Redis is running: `redis-cli ping` (should return `PONG`)
2. Check `REDIS_HOST` and `REDIS_PORT` in `.env`
3. If Redis is remote, ensure network connectivity: `ping <REDIS_HOST>`

### "ModuleNotFoundError: No module named 'redis'"

**Symptom:** Python can't find the redis module

**Fix:**
```bash
pip install redis>=4.5.0
# Or reinstall all deps:
pip install -r requirements.txt --force-reinstall
```

### "No such file or directory: ./cartridges"

**Symptom:** Orchestrator can't find cartridge directory

**Fix:**
```bash
mkdir -p $CARTRIDGE_DIR
# Or set CARTRIDGE_DIR to an existing path in .env
```

### "RuntimeError: CUDA out of memory" (MTR initialization)

**Symptom:** PyTorch can't allocate GPU memory for MTR

**Fix:**
```bash
# Force CPU-only mode (slower but always works)
export TORCH_DEVICE=cpu

# Or reduce grain cache size
export GRAIN_CACHE_SIZE=5000
```

### "ImportError: cannot import name 'QueryOrchestrator'"

**Symptom:** Python can't find Kitbash modules

**Fix:**
1. Verify you're in the `kitbash/` directory: `pwd`
2. Add to Python path if necessary:
   ```bash
   export PYTHONPATH=$(pwd):$PYTHONPATH
   python3 -c "from query_orchestrator_posix import QueryOrchestrator"
   ```

---

## Next Steps

Once setup is complete:

1. **Read ARCHITECTURE.md** — Understand the query flow and component interactions
2. **Read CARTRIDGE.md** — Learn how to build or inspect cartridges
3. **Run tests** — `python3 -m pytest tests/` (when test suite is available)
4. **Explore examples** — Check `examples/` for sample queries and workflows

---

## Environment Checklist

- [ ] Python 3.14+ installed
- [ ] Redis running (`redis-cli ping` returns PONG)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with configuration
- [ ] `CARTRIDGE_DIR` exists and is accessible
- [ ] First query runs without error
- [ ] `kitbash.log` is being written to

If all checks pass, you're ready to develop.
