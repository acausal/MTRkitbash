"""
src/redis_coupling.py - Coupling Validation Wrapper

Python interface for Redis Lua coupling validation scripts.
Bridges between QueryOrchestrator (Python) and Redis (Lua).

Phase 3B.3: Coupling Geometry Implementation
"""

import json
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import redis
from redis import Redis

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CouplingDelta:
    """Result of a coupling validation between two epistemic layers."""
    query_id: str
    layer_a: str  # "L0", "L1", "L2", "L3", "L4", "L5"
    layer_b: str
    status: str  # "OK", "FLAG", "FAIL"
    delta_magnitude: float  # 0.0 to 1.0 (how bad is the mismatch)
    severity: str  # "PASS", "LOW", "MEDIUM", "HIGH", "CRITICAL"
    coupling_constant: float  # κ value used
    timestamp: int  # Unix timestamp when checked
    fact_a_id: Optional[str] = None  # ID of fact in layer_a (if applicable)
    fact_b_id: Optional[str] = None  # ID of fact in layer_b (if applicable)
    reasoning: str = ""  # Human-readable explanation

    def to_json(self) -> str:
        """Serialize to JSON for Redis storage."""
        return json.dumps({
            "query_id": self.query_id,
            "layer_a": self.layer_a,
            "layer_b": self.layer_b,
            "status": self.status,
            "delta_magnitude": self.delta_magnitude,
            "severity": self.severity,
            "coupling_constant": self.coupling_constant,
            "timestamp": self.timestamp,
            "fact_a_id": self.fact_a_id,
            "fact_b_id": self.fact_b_id,
            "reasoning": self.reasoning,
        })

    @staticmethod
    def from_json(data: str) -> "CouplingDelta":
        """Deserialize from JSON."""
        obj = json.loads(data)
        return CouplingDelta(**obj)


# ============================================================================
# COUPLING VALIDATOR
# ============================================================================

class CouplingValidator:
    """
    Validates coupling between epistemic layers using Redis Lua scripts.
    
    Attributes:
        redis_client: Redis connection
        script_validate: Lua script for validate_coupling
        script_record: Lua script for record_coupling_delta
    """

    def __init__(self, redis_client: Redis):
        """
        Initialize coupling validator.
        
        Args:
            redis_client: Connected Redis client
            
        Raises:
            ConnectionError: If Redis not available
        """
        self.redis_client = redis_client
        self.script_validate = None
        self.script_record = None
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("✓ Coupling validator connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"✗ Failed to connect to Redis: {e}")
            raise

    def register_scripts(self, lua_script_path: str) -> bool:
        """
        Load and register Lua scripts from file.
        
        Args:
            lua_script_path: Path to redis_coupling_scripts.lua
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(lua_script_path, 'r') as f:
                lua_code = f.read()
            
            # For MVP: Register two main scripts
            # (Full implementation would parse the Lua file structure)
            
            # Script 1: validate_coupling
            validate_script = """
            local query_id = KEYS[1]
            local layer_a = ARGV[1]
            local layer_b = ARGV[2]
            local coupling_constant = tonumber(ARGV[3]) or 1.0
            
            -- Placeholder: Return basic validation result
            -- Real implementation calls sub-functions from lua_code
            
            local result = {
                status = "OK",
                delta = 0.0,
                severity = "LOW",
                layer_a = layer_a,
                layer_b = layer_b,
                query_id = query_id,
                coupling_constant = coupling_constant,
                timestamp = redis.call("TIME")[1]
            }
            
            return cjson.encode(result)
            """
            
            # Script 2: record_coupling_delta
            record_script = """
            local query_id = KEYS[1]
            local delta_json = ARGV[1]
            
            local deltas_key = "query:" .. query_id .. ":deltas"
            redis.call("LPUSH", deltas_key, delta_json)
            
            -- Keep TTL in sync
            local ttl = redis.call("TTL", "query:" .. query_id .. ":metadata")
            if ttl > 0 then
                redis.call("EXPIRE", deltas_key, ttl)
            end
            
            return "OK"
            """
            
            # Register scripts
            self.script_validate = self.redis_client.register_script(validate_script)
            self.script_record = self.redis_client.register_script(record_script)
            
            logger.info("✓ Coupling validation scripts registered")
            return True
            
        except FileNotFoundError:
            logger.error(f"✗ Script file not found: {lua_script_path}")
            return False
        except Exception as e:
            logger.error(f"✗ Failed to register scripts: {e}")
            return False

    def validate_coupling(
        self,
        query_id: str,
        layer_a: str,
        layer_b: str,
        coupling_constant: float = 1.0
    ) -> Optional[CouplingDelta]:
        """
        Validate coupling between two epistemic layers.
        
        Args:
            query_id: Query ID (Redis key prefix)
            layer_a: First layer ("L0", "L1", etc.)
            layer_b: Second layer
            coupling_constant: κ value (rigidity tuning)
            
        Returns:
            CouplingDelta object, or None if validation fails
        """
        if not self.script_validate:
            logger.warning("Validation scripts not registered")
            return None
        
        try:
            # Run Lua script (atomic in Redis)
            result = self.script_validate(
                keys=[query_id],
                args=[layer_a, layer_b, coupling_constant]
            )
            
            # Parse result (JSON from Lua)
            delta_dict = json.loads(result.decode() if isinstance(result, bytes) else result)
            
            # Convert to CouplingDelta object
            delta = CouplingDelta(
                query_id=query_id,
                layer_a=delta_dict.get("layer_a", layer_a),
                layer_b=delta_dict.get("layer_b", layer_b),
                status=delta_dict.get("status", "UNKNOWN"),
                delta_magnitude=delta_dict.get("delta", 0.0),
                severity=delta_dict.get("severity", "LOW"),
                coupling_constant=coupling_constant,
                timestamp=int(delta_dict.get("timestamp", 0)),
            )
            
            logger.debug(f"Coupling validation: {layer_a} vs {layer_b} = {delta.severity}")
            
            return delta
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse validation result: {e}")
            return None
        except Exception as e:
            logger.error(f"Coupling validation failed: {e}")
            return None

    def record_delta(self, delta: CouplingDelta) -> bool:
        """
        Record a coupling delta in Redis.
        
        Args:
            delta: CouplingDelta to record
            
        Returns:
            True if recorded successfully
        """
        if not self.script_record:
            logger.warning("Recording scripts not registered")
            return False
        
        try:
            # Run Lua script to record the delta
            result = self.script_record(
                keys=[delta.query_id],
                args=[delta.to_json()]
            )
            
            logger.debug(f"Delta recorded: {delta.layer_a} vs {delta.layer_b}")
            return result == "OK" or result == b"OK"
            
        except Exception as e:
            logger.error(f"Failed to record delta: {e}")
            return False

    def validate_and_record(
        self,
        query_id: str,
        layer_a: str,
        layer_b: str,
        coupling_constant: float = 1.0
    ) -> Optional[CouplingDelta]:
        """
        Validate coupling AND record the result in one operation.
        
        Args:
            query_id: Query ID
            layer_a: First layer
            layer_b: Second layer
            coupling_constant: κ value
            
        Returns:
            CouplingDelta object (including recorded delta), or None
        """
        delta = self.validate_coupling(query_id, layer_a, layer_b, coupling_constant)
        
        if delta:
            self.record_delta(delta)
        
        return delta

    def get_deltas_for_query(self, query_id: str) -> list:
        """
        Retrieve all coupling deltas for a query.
        
        Args:
            query_id: Query ID
            
        Returns:
            List of CouplingDelta objects
        """
        try:
            deltas_key = f"query:{query_id}:deltas"
            delta_jsons = self.redis_client.lrange(deltas_key, 0, -1)
            
            deltas = [
                CouplingDelta.from_json(d.decode() if isinstance(d, bytes) else d)
                for d in delta_jsons
            ]
            
            return deltas
            
        except Exception as e:
            logger.error(f"Failed to retrieve deltas: {e}")
            return []

    def get_severity_summary(self, query_id: str) -> Dict[str, int]:
        """
        Get count of deltas by severity.
        
        Args:
            query_id: Query ID
            
        Returns:
            Dict with severity counts
        """
        deltas = self.get_deltas_for_query(query_id)
        
        summary = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
        }
        
        for delta in deltas:
            if delta.severity in summary:
                summary[delta.severity] += 1
        
        return summary

    def has_critical_violations(self, query_id: str) -> bool:
        """
        Check if query has any CRITICAL coupling violations.
        
        Args:
            query_id: Query ID
            
        Returns:
            True if any CRITICAL violations exist
        """
        deltas = self.get_deltas_for_query(query_id)
        return any(d.severity == "CRITICAL" for d in deltas)

    def has_high_violations(self, query_id: str) -> bool:
        """
        Check if query has any HIGH severity violations.
        
        Args:
            query_id: Query ID
            
        Returns:
            True if any HIGH violations exist
        """
        deltas = self.get_deltas_for_query(query_id)
        return any(d.severity == "HIGH" for d in deltas)


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def create_coupling_validator(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0
) -> Optional[CouplingValidator]:
    """
    Factory function to create and initialize a CouplingValidator.
    
    Args:
        host: Redis host
        port: Redis port
        db: Redis database number
        
    Returns:
        CouplingValidator instance, or None if initialization fails
    """
    try:
        r = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=2
        )
        
        validator = CouplingValidator(r)
        logger.info("✓ Coupling validator initialized")
        return validator
        
    except Exception as e:
        logger.error(f"✗ Failed to create coupling validator: {e}")
        return None


# ============================================================================
# TESTING & DEBUGGING
# ============================================================================

if __name__ == "__main__":
    """Quick test of coupling validator."""
    
    logging.basicConfig(level=logging.DEBUG)
    
    # Create validator
    validator = create_coupling_validator()
    
    if validator:
        # Register scripts
        validator.register_scripts("redis_coupling_scripts.lua")
        
        # Test validation
        test_query_id = "test_query_001"
        
        print("\n=== Testing Coupling Validation ===\n")
        
        # Test 1: L0 vs L1
        print("Test 1: Validating L0 vs L1")
        delta = validator.validate_and_record(test_query_id, "L0", "L1")
        if delta:
            print(f"  Result: {delta.severity} (delta={delta.delta_magnitude})")
        
        # Test 2: L1 vs L2
        print("\nTest 2: Validating L1 vs L2")
        delta = validator.validate_and_record(test_query_id, "L1", "L2")
        if delta:
            print(f"  Result: {delta.severity} (delta={delta.delta_magnitude})")
        
        # Test 3: L2 vs L4
        print("\nTest 3: Validating L2 vs L4")
        delta = validator.validate_and_record(test_query_id, "L2", "L4")
        if delta:
            print(f"  Result: {delta.severity} (delta={delta.delta_magnitude})")
        
        # Get summary
        print("\n=== Query Summary ===\n")
        summary = validator.get_severity_summary(test_query_id)
        print(f"Summary: {summary}")
        
        has_critical = validator.has_critical_violations(test_query_id)
        has_high = validator.has_high_violations(test_query_id)
        print(f"Critical violations: {has_critical}")
        print(f"High violations: {has_high}")
        
        print("\n✓ Coupling validator tests complete")
    else:
        print("✗ Failed to initialize validator")
