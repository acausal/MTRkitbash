"""
Redis Blackboard - Shared state coordination for Kitbash orchestration.

Provides abstraction layer for Redis operations used in Phase 3B.
All workers communicate through Redis without direct RPC calls.

Schema:
  kitbash:grains:<fact_id> -> JSON grain data
  kitbash:queries:queue -> list of pending query IDs
  kitbash:queries:state:<query_id> -> JSON query state
  kitbash:diagnostic:feed -> list of diagnostic events
  kitbash:health:<worker_name> -> JSON health status
  kitbash:metrics:<metric_name> -> metric data (sorted sets)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
import redis
from redis import Redis

logger = logging.getLogger(__name__)


class RedisBlackboard:
    """
    Redis-backed shared blackboard for Kitbash orchestration.

    Handles:
    - Query state storage and retrieval
    - Grain index management
    - Diagnostic event logging
    - Worker health tracking
    - Metrics collection
    """

    def __init__(self, redis_client: Optional[Redis] = None, prefix: str = "kitbash:"):
        """
        Initialize Redis blackboard.

        Args:
            redis_client: Redis connection. If None, creates new connection to localhost.
            prefix: Redis key prefix for all operations.
        """
        self.redis = redis_client or redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True
        )
        self.prefix = prefix

        # Validate Redis connection
        try:
            self.redis.ping()
            logger.info("Redis connection established")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    # Query State Management

    def create_query(self, query_id: str, query_text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a new query state in Redis.

        Args:
            query_id: Unique query identifier
            query_text: The query text
            metadata: Optional metadata dict
        """
        query_state = {
            "query_id": query_id,
            "query_text": query_text,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "layer_attempts": [],
            "metadata": metadata or {},
        }
        key = f"{self.prefix}queries:state:{query_id}"
        self.redis.set(key, json.dumps(query_state))
        logger.debug(f"Created query state: {query_id}")

    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve query state from Redis.

        Args:
            query_id: Query identifier

        Returns:
            Query state dict or None if not found
        """
        key = f"{self.prefix}queries:state:{query_id}"
        data = self.redis.get(key)
        return json.loads(data) if data else None

    def update_query_status(
        self,
        query_id: str,
        status: str,
        layer_result: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update query status and add layer attempt.

        Args:
            query_id: Query identifier
            status: New status (pending, layer0_hit, layer1_attempted, etc.)
            layer_result: Result from layer processing
        """
        query_state = self.get_query(query_id)
        if not query_state:
            logger.warning(f"Query not found: {query_id}")
            return

        query_state["status"] = status

        if status == "started":
            query_state["started_at"] = datetime.now().isoformat()
        elif status == "completed":
            query_state["completed_at"] = datetime.now().isoformat()

        if layer_result:
            query_state["layer_attempts"].append({
                "timestamp": datetime.now().isoformat(),
                "result": layer_result,
            })

        key = f"{self.prefix}queries:state:{query_id}"
        self.redis.set(key, json.dumps(query_state))

    def delete_query(self, query_id: str) -> None:
        """Delete query state from Redis."""
        key = f"{self.prefix}queries:state:{query_id}"
        self.redis.delete(key)

    # Query Queue Management

    def enqueue_query(self, query_id: str) -> None:
        """Add query to processing queue."""
        queue_key = f"{self.prefix}queries:queue"
        self.redis.lpush(queue_key, query_id)
        logger.debug(f"Enqueued query: {query_id}")

    def dequeue_query(self) -> Optional[str]:
        """Pop query from processing queue."""
        queue_key = f"{self.prefix}queries:queue"
        query_id = self.redis.rpop(queue_key)
        return query_id

    def queue_length(self) -> int:
        """Get number of queries in queue."""
        queue_key = f"{self.prefix}queries:queue"
        return self.redis.llen(queue_key)

    def peek_queue(self, count: int = 10) -> List[str]:
        """Peek at next N queries without removing them."""
        queue_key = f"{self.prefix}queries:queue"
        return self.redis.lrange(queue_key, 0, count - 1)

    # Grain Index Management

    def store_grain(self, fact_id: str, grain_data: Dict[str, Any]) -> None:
        """
        Store a grain in the Redis index.

        Args:
            fact_id: Fact identifier (used as key)
            grain_data: Grain data (dict with ternary representation)
        """
        key = f"{self.prefix}grains:{fact_id}"
        self.redis.hset(key, "data", json.dumps(grain_data))
        self.redis.hset(key, "updated_at", datetime.now().isoformat())
        logger.debug(f"Stored grain: {fact_id}")

    def get_grain(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve grain from index."""
        key = f"{self.prefix}grains:{fact_id}"
        data = self.redis.hget(key, "data")
        return json.loads(data) if data else None

    def grain_exists(self, fact_id: str) -> bool:
        """Check if grain exists in index."""
        key = f"{self.prefix}grains:{fact_id}"
        return self.redis.exists(key) > 0

    def list_grains(self, pattern: str = "*") -> List[str]:
        """List all grain IDs matching pattern."""
        full_pattern = f"{self.prefix}grains:{pattern}"
        keys = self.redis.keys(full_pattern)
        return [k.replace(f"{self.prefix}grains:", "") for k in keys]

    def grain_count(self) -> int:
        """Get total number of stored grains."""
        pattern = f"{self.prefix}grains:*"
        return len(self.redis.keys(pattern))

    # Diagnostic Event Logging

    def log_diagnostic_event(
        self,
        event_type: str,
        query_id: str,
        details: Dict[str, Any]
    ) -> None:
        """
        Log a diagnostic event to Redis.

        Args:
            event_type: Type of event (layer_attempt, timeout, escalation, etc.)
            query_id: Associated query ID
            details: Event details
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "query_id": query_id,
            "details": details,
        }
        feed_key = f"{self.prefix}diagnostic:feed"
        self.redis.lpush(feed_key, json.dumps(event))

        # Trim to last 10000 events
        self.redis.ltrim(feed_key, 0, 9999)

    def get_diagnostic_feed(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent diagnostic events."""
        feed_key = f"{self.prefix}diagnostic:feed"
        events = self.redis.lrange(feed_key, 0, count - 1)
        return [json.loads(e) for e in events]

    def clear_diagnostic_feed(self) -> None:
        """Clear all diagnostic events."""
        feed_key = f"{self.prefix}diagnostic:feed"
        self.redis.delete(feed_key)

    # Worker Health Tracking

    def set_worker_health(self, worker_name: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Update worker health status.

        Args:
            worker_name: Name of worker (bitnet, cartridge, kobold)
            status: Health status (healthy, degraded, dead)
            details: Additional health details
        """
        health_data = {
            "worker_name": worker_name,
            "status": status,
            "last_heartbeat": datetime.now().isoformat(),
            "details": details or {},
        }
        key = f"{self.prefix}health:{worker_name}"
        self.redis.set(key, json.dumps(health_data), ex=300)  # 5 minute expiry

    def get_worker_health(self, worker_name: str) -> Optional[Dict[str, Any]]:
        """Get worker health status."""
        key = f"{self.prefix}health:{worker_name}"
        data = self.redis.get(key)
        return json.loads(data) if data else None

    def all_workers_healthy(self) -> Dict[str, str]:
        """Get health status of all workers."""
        pattern = f"{self.prefix}health:*"
        keys = self.redis.keys(pattern)
        health_status = {}
        for key in keys:
            worker_name = key.replace(f"{self.prefix}health:", "")
            data = self.redis.get(key)
            if data:
                status = json.loads(data)
                health_status[worker_name] = status.get("status", "unknown")
            else:
                health_status[worker_name] = "unknown"
        return health_status

    # Metrics Collection

    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """
        Record a metric value with timestamp.

        Args:
            metric_name: Name of metric
            value: Numeric value
            timestamp: Timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        score = timestamp.timestamp()
        key = f"{self.prefix}metrics:{metric_name}"
        self.redis.zadd(key, {str(value): score})

    def get_metrics(self, metric_name: str, minutes: int = 60) -> List[float]:
        """
        Get metric values from last N minutes.

        Args:
            metric_name: Name of metric
            minutes: Number of minutes to look back

        Returns:
            List of metric values (oldest first)
        """
        cutoff_time = (datetime.now() - timedelta(minutes=minutes)).timestamp()
        key = f"{self.prefix}metrics:{metric_name}"
        values = self.redis.zrangebyscore(key, cutoff_time, "+inf")
        return [float(v) for v in values]

    def get_metric_percentile(self, metric_name: str, percentile: float, minutes: int = 60) -> Optional[float]:
        """
        Calculate metric percentile from recent values.

        Args:
            metric_name: Name of metric
            percentile: Percentile (0-100)
            minutes: Number of minutes to look back

        Returns:
            Percentile value or None if insufficient data
        """
        values = self.get_metrics(metric_name, minutes)
        if not values:
            return None

        sorted_values = sorted(values)
        idx = int(len(sorted_values) * (percentile / 100))
        return sorted_values[idx]

    # Cleanup & Maintenance

    def cleanup_old_queries(self, hours: int = 24) -> int:
        """
        Delete old query states.

        Args:
            hours: Delete queries older than this many hours

        Returns:
            Number of queries deleted
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        pattern = f"{self.prefix}queries:state:*"
        keys = self.redis.keys(pattern)

        deleted = 0
        for key in keys:
            data = self.redis.get(key)
            if data:
                query_state = json.loads(data)
                created_at = datetime.fromisoformat(query_state.get("created_at", ""))
                if created_at < cutoff_time:
                    self.redis.delete(key)
                    deleted += 1

        return deleted

    def flush_all(self) -> None:
        """Clear all Kitbash data from Redis (development only)."""
        pattern = f"{self.prefix}*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
            logger.info(f"Flushed {len(keys)} keys from Redis")

    # Utility Methods

    def redis_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        return self.redis.info()

    def key_count(self) -> int:
        """Get total key count under Kitbash prefix."""
        pattern = f"{self.prefix}*"
        return len(self.redis.keys(pattern))

    def close(self) -> None:
        """Close Redis connection."""
        self.redis.close()
        logger.info("Redis connection closed")


if __name__ == "__main__":
    # Test the blackboard
    logging.basicConfig(level=logging.DEBUG)

    try:
        bb = RedisBlackboard()

        # Test query operations
        bb.create_query("test_query_1", "What is the capital of France?")
        query = bb.get_query("test_query_1")
        print(f"Created query: {query}")

        # Test queue operations
        bb.enqueue_query("test_query_1")
        print(f"Queue length: {bb.queue_length()}")

        # Test grain operations
        bb.store_grain("fact_001", {"ternary": [1, -1, 0, 1], "meaning": "Paris is France capital"})
        grain = bb.get_grain("fact_001")
        print(f"Stored grain: {grain}")

        # Test diagnostic logging
        bb.log_diagnostic_event("layer0_hit", "test_query_1", {"confidence": 0.99})
        events = bb.get_diagnostic_feed(5)
        print(f"Diagnostic events: {events}")

        # Test health tracking
        bb.set_worker_health("bitnet", "healthy", {"load": 0.5})
        health = bb.all_workers_healthy()
        print(f"Worker health: {health}")

        # Test metrics
        bb.record_metric("layer0_latency_ms", 0.17)
        bb.record_metric("layer0_latency_ms", 0.19)
        metrics = bb.get_metrics("layer0_latency_ms", minutes=60)
        print(f"Metrics: {metrics}")

        print("\n✅ All blackboard tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()