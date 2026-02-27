"""
kitbash/logging/structured_logger.py

Structured, async-ready logging for Kitbash.
Events are JSON objects written to JSONL files organized by category and date.

Core components:
- LogEvent: dataclass for all events
- EventWriter: thread-safe buffered writer to disk
- ComponentLogger: simple interface for components
- get_event_logger(): global factory function

Usage:
    logger = get_event_logger("query_orchestrator")
    logger.log(
        event_type="query_received",
        data={"query_text": "...", "tokens": 5},
        query_id="q_abc123",
        category="query_lifecycle"
    )

Threading model:
    - Main thread: create events, queue them (non-blocking)
    - Background thread: flush periodically to disk
    - Never blocks query processing

Future: asyncio migration is straightforward (swap queue.Queue for asyncio.Queue)
"""

import json
import queue
import threading
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


# ============================================================================
# Log Level Enumeration
# ============================================================================

class LogLevel(Enum):
    """Standard log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ============================================================================
# Event Dataclass
# ============================================================================

@dataclass
class LogEvent:
    """
    Base log event. Self-describing and JSON-serializable.
    
    Fields:
        timestamp: ISO 8601 with microseconds (auto-filled)
        event_type: Semantic label (e.g., "query_received", "layer_attempt")
        component: Which component logged this
        category: Category for filtering (query_lifecycle, layer_execution, etc.)
        level: Severity (debug, info, warning, error)
        data: Event-specific fields (the actual content)
        query_id: Optional, links events in same query lifecycle
        metadata: Optional context (session_id, turn_number, etc.)
    """
    timestamp: str
    event_type: str
    component: str
    category: str
    level: str
    data: Dict[str, Any]
    query_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)


def make_event(
    event_type: str,
    component: str,
    category: str,
    data: Dict[str, Any],
    query_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    level: str = "info"
) -> LogEvent:
    """
    Create a log event with auto-filled timestamp.
    
    Args:
        event_type: What happened (e.g., "query_received")
        component: Which component (e.g., "query_orchestrator")
        category: Category for filtering (e.g., "query_lifecycle")
        data: Event-specific fields
        query_id: Optional, links to query lifecycle
        metadata: Optional context
        level: Severity level
    
    Returns:
        LogEvent ready to log
    """
    return LogEvent(
        timestamp=datetime.utcnow().isoformat() + "Z",
        event_type=event_type,
        component=component,
        category=category,
        level=level,
        data=data,
        query_id=query_id,
        metadata=metadata
    )


# ============================================================================
# Event Writer
# ============================================================================

class EventWriter:
    """
    Thread-safe event writer with buffering and async flushing.
    
    Features:
    - Queues events from multiple threads without blocking
    - Buffers events by category
    - Writes JSONL to ~/.kitbash/logs/{category}/YYYY-MM-DD.jsonl
    - Async-ready (threading.Thread now, asyncio.Task later)
    - Toggleable by category
    - Graceful failure on queue overflow
    
    Performance:
    - Queue insertion: <0.02ms
    - Flushing happens on background thread (non-blocking)
    - No impact on query processing
    """
    
    LOG_DIR = Path.home() / ".kitbash" / "logs"
    BUFFER_SIZE = 100              # Flush after N events
    FLUSH_INTERVAL_S = 1.0         # Or after T seconds
    MAX_QUEUE_SIZE = 5000          # Drop if queue full
    
    def __init__(self, enabled: bool = True):
        """
        Initialize event writer.
        
        Args:
            enabled: Master toggle for all logging
        """
        self.enabled = enabled
        self.enabled_categories: Set[str] = set()  # Empty set = all enabled
        self.log_dir = self.LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Event queue (thread-safe)
        self.event_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        
        # Buffers per category (protected by lock)
        self.buffers: Dict[str, List[LogEvent]] = {}
        self.buffer_lock = threading.Lock()
        
        # Standard logger for errors
        self._logger = logging.getLogger("kitbash.logging")
        
        # Background flush thread
        self.flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="KitbashLogFlush"
        )
        self.flush_thread.start()
    
    def log_event(self, event: LogEvent) -> bool:
        """
        Queue an event for logging.
        
        Args:
            event: LogEvent to queue
        
        Returns:
            True if queued successfully, False if dropped (queue full)
        """
        if not self.enabled:
            return False
        
        # Check if category is enabled (empty set = all enabled)
        if self.enabled_categories and event.category not in self.enabled_categories:
            return False
        
        try:
            self.event_queue.put(event, block=False)
            return True
        except queue.Full:
            # Log overflow warning to stderr and continue
            self._logger.warning(
                f"Log queue full, dropped event: {event.event_type} "
                f"from {event.component}"
            )
            return False
    
    def toggle_all(self, enabled: bool):
        """
        Master toggle for all logging.
        
        Args:
            enabled: True to enable all categories, False to disable all
        """
        self.enabled = enabled
    
    def set_enabled_categories(self, categories: Optional[List[str]]):
        """
        Set which categories to log.
        
        Args:
            categories: List of category names to enable
                       None or empty list = all enabled
        """
        if categories:
            self.enabled_categories = set(categories)
        else:
            self.enabled_categories = set()
    
    def _flush_loop(self):
        """
        Background thread: consume events and flush periodically.
        Runs forever (daemon thread).
        """
        last_flush = time.time()
        
        while True:
            try:
                # Try to get one event (non-blocking timeout)
                try:
                    event = self.event_queue.get(timeout=0.1)
                    with self.buffer_lock:
                        if event.category not in self.buffers:
                            self.buffers[event.category] = []
                        self.buffers[event.category].append(event)
                
                except queue.Empty:
                    pass
                
                # Check if should flush
                now = time.time()
                should_flush = False
                
                with self.buffer_lock:
                    total_events = sum(len(v) for v in self.buffers.values())
                    if total_events >= self.BUFFER_SIZE:
                        should_flush = True
                    elif now - last_flush >= self.FLUSH_INTERVAL_S and total_events > 0:
                        should_flush = True
                
                if should_flush:
                    self._do_flush()
                    last_flush = now
            
            except Exception as e:
                self._logger.error(f"Log flush loop error: {e}")
                time.sleep(0.1)
    
    def _do_flush(self):
        """Write all buffered events to disk, organized by category and date."""
        with self.buffer_lock:
            if not self.buffers:
                return
            
            today = datetime.utcnow().strftime("%Y-%m-%d")
            
            for category, events in self.buffers.items():
                if not events:
                    continue
                
                try:
                    # Path: ~/.kitbash/logs/{category}/{YYYY-MM-DD}.jsonl
                    cat_dir = self.log_dir / category
                    cat_dir.mkdir(parents=True, exist_ok=True)
                    
                    log_file = cat_dir / f"{today}.jsonl"
                    
                    # Append events (jsonl format = one JSON object per line)
                    with open(log_file, "a") as f:
                        for event in events:
                            f.write(json.dumps(event.to_dict()) + "\n")
                
                except Exception as e:
                    self._logger.error(f"Failed to flush logs for {category}: {e}")
            
            # Clear buffers after successful flush
            self.buffers = {}


# ============================================================================
# Global Writer Instance
# ============================================================================

_event_writer: Optional[EventWriter] = None


def get_event_writer(enabled: bool = True) -> EventWriter:
    """
    Get or initialize the global event writer.
    
    Args:
        enabled: Initial enabled state (only used on first call)
    
    Returns:
        Global EventWriter instance
    """
    global _event_writer
    if _event_writer is None:
        _event_writer = EventWriter(enabled=enabled)
    return _event_writer


# ============================================================================
# Component Logger
# ============================================================================

class ComponentLogger:
    """
    Simple interface for components to log events.
    
    Usage:
        logger = get_event_logger("my_component")
        logger.log(
            event_type="something_happened",
            data={"field": "value"},
            category="query_lifecycle",
            query_id="q_123"
        )
    
    Convenience methods:
        logger.debug(event_type, data, ...)
        logger.warning(event_type, data, ...)
        logger.error(event_type, data, ...)
    """
    
    def __init__(self, component: str, writer: EventWriter):
        """
        Initialize component logger.
        
        Args:
            component: Component name (e.g., "query_orchestrator")
            writer: EventWriter instance
        """
        self.component = component
        self.writer = writer
    
    def log(
        self,
        event_type: str,
        data: Dict[str, Any],
        category: str = "component_stats",
        query_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: str = "info"
    ) -> bool:
        """
        Log an event.
        
        Args:
            event_type: What happened (e.g., "query_received")
            data: Event-specific fields
            category: Category for filtering (query_lifecycle, layer_execution, etc.)
            query_id: Optional, links to query lifecycle
            metadata: Optional context (session_id, turn_number, etc.)
            level: Severity level (debug, info, warning, error)
        
        Returns:
            True if logged, False if dropped (queue full)
        """
        event = make_event(
            event_type=event_type,
            component=self.component,
            category=category,
            data=data,
            query_id=query_id,
            metadata=metadata,
            level=level
        )
        return self.writer.log_event(event)
    
    # Convenience methods for different severity levels
    def debug(self, event_type: str, data: Dict[str, Any], **kwargs) -> bool:
        """Log a debug-level event."""
        return self.log(event_type, data, level="debug", **kwargs)
    
    def warning(self, event_type: str, data: Dict[str, Any], **kwargs) -> bool:
        """Log a warning-level event."""
        return self.log(event_type, data, level="warning", **kwargs)
    
    def error(self, event_type: str, data: Dict[str, Any], **kwargs) -> bool:
        """Log an error-level event."""
        return self.log(event_type, data, level="error", **kwargs)


def get_event_logger(component_name: str) -> ComponentLogger:
    """
    Get or create a component logger.
    
    Args:
        component_name: Name of component (e.g., "query_orchestrator")
    
    Returns:
        ComponentLogger instance
    """
    writer = get_event_writer()
    return ComponentLogger(component_name, writer)


# ============================================================================
# Testing & Utilities
# ============================================================================

def test_logging_integration():
    """
    Test that logging works without breaking queries.
    """
    import json
    
    # Initialize
    writer = get_event_writer(enabled=True)
    logger = get_event_logger("test_component")
    
    # Log some test events
    logger.log(
        event_type="test_event_1",
        data={"test_field": "test_value_1"},
        category="query_lifecycle"
    )
    
    logger.log(
        event_type="test_event_2",
        data={"test_field": "test_value_2"},
        category="layer_execution"
    )
    
    # Force flush
    writer._do_flush()
    
    # Check files were created
    log_dir = Path.home() / ".kitbash" / "logs"
    assert (log_dir / "query_lifecycle").exists(), "query_lifecycle directory not created"
    assert (log_dir / "layer_execution").exists(), "layer_execution directory not created"
    
    today = datetime.utcnow().strftime("%Y-%m-%d")
    query_log_file = log_dir / "query_lifecycle" / f"{today}.jsonl"
    layer_log_file = log_dir / "layer_execution" / f"{today}.jsonl"
    
    assert query_log_file.exists(), f"Log file not created: {query_log_file}"
    assert layer_log_file.exists(), f"Log file not created: {layer_log_file}"
    
    # Check content
    with open(query_log_file) as f:
        lines = f.readlines()
        assert len(lines) > 0, "No events written to query_lifecycle"
        event = json.loads(lines[-1])
        assert event["event_type"] == "test_event_1"
        assert event["component"] == "test_component"
    
    with open(layer_log_file) as f:
        lines = f.readlines()
        assert len(lines) > 0, "No events written to layer_execution"
        event = json.loads(lines[-1])
        assert event["event_type"] == "test_event_2"
    
    print("✅ Logging integration test passed")
    print(f"   Query lifecycle log: {query_log_file}")
    print(f"   Layer execution log: {layer_log_file}")
    return True


if __name__ == "__main__":
    test_logging_integration()
