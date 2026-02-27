"""
BitNet Engine - Phase 3B Layer 1 Wrapper

Wraps the BitNet HTTP server (ternary neural network inference)
in the InferenceEngine interface for use by QueryOrchestrator.

Provides fast, learned inference using 1.58-bit quantized weights.
Falls between crystallized grain lookup (fast/narrow) and full reasoning (slow/broad).

Used by: QueryOrchestrator (first or second layer)
Wraps: llama-server.exe running BitNet model (external process)
Interface: HTTP POST to /completion endpoint
"""

import time
import logging
import requests
from typing import Optional, Dict, Any
from urllib.parse import urljoin

from interfaces.inference_engine import InferenceEngine, InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)


class BitNetEngine(InferenceEngine):
    """
    Layer 1 Inference Engine - BitNet Ternary Network.
    
    Wraps the BitNet HTTP server for fast ternary neural network inference.
    
    Performance:
    - Setup: 0ms (connects to running server)
    - Query latency: 2-6 seconds per 100 tokens (on GTX 1060)
    - Model: Little-Bitch-3B.i1-Q6_K (3B parameters, 1.58-bit quantized)
    - Confidence: 0.70-0.80 (learned, not crystallized)
    
    Architecture:
    - Expects BitNet server running at http://127.0.0.1:8080
    - POSTs query to /completion endpoint
    - Parses JSON response with answer and metadata
    - Handles timeouts and connection failures gracefully
    
    Server Startup:
        llama-server.exe --model <path/to/model.gguf> -ngl 20 -c 512
    """
    
    engine_name = "BITNET"
    
    def __init__(self, server_url: str = "http://127.0.0.1:8080",
                 timeout_seconds: int = 30,
                 max_tokens: int = 100,
                 temperature: float = 0.7):
        """
        Initialize BitNet Engine.
        
        Args:
            server_url: URL of BitNet HTTP server
            timeout_seconds: Request timeout
            max_tokens: Maximum tokens to generate per query
            temperature: Sampling temperature (0.0-2.0)
        
        Raises:
            ValueError: If parameters invalid
        """
        super().__init__()
        
        self.server_url = server_url.rstrip('/')
        self.completion_endpoint = urljoin(self.server_url, '/completion')
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Statistics
        self.query_count = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.total_latency_ms = 0.0
        self.total_tokens_generated = 0
        
        # Validation
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0], got {temperature}")
        
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        
        # Try to verify server is running
        self._verify_server()
    
    def _verify_server(self) -> None:
        """
        Verify BitNet server is running and responsive.
        
        Logs warning if unreachable but doesn't fail (server might be slow to start).
        """
        try:
            response = requests.head(
                self.server_url,
                timeout=5,
            )
            logger.info(f"✓ BitNet server reachable at {self.server_url}")
        
        except requests.ConnectionError:
            logger.warning(
                f"BitNet server not reachable at {self.server_url} - "
                f"make sure llama-server.exe is running"
            )
        
        except Exception as e:
            logger.warning(f"Could not verify BitNet server: {e}")
    
    def is_available(self) -> bool:
        """
        Check if BitNet server is reachable.
        
        Quick check - doesn't block on slow server.
        """
        try:
            response = requests.head(
                self.server_url,
                timeout=2,
            )
            return response.status_code < 500
        
        except Exception:
            return False
    
    def query(self, request: InferenceRequest) -> InferenceResponse:
        """
        Execute a BitNet query via HTTP.
        
        Args:
            request: InferenceRequest with user query
        
        Returns:
            InferenceResponse with BitNet's answer
        
        Raises:
            RuntimeError: If server unreachable or request fails
        """
        self.query_count += 1
        start_time = time.perf_counter()
        
        try:
            # Prepare request
            payload = {
                "prompt": request.user_query,
                "n_predict": self.max_tokens,
                "temperature": self.temperature,
            }
            
            # Send to BitNet server
            logger.debug(f"Sending query to BitNet: {request.user_query[:50]}...")
            
            response = requests.post(
                self.completion_endpoint,
                json=payload,
                timeout=self.timeout_seconds,
            )
            
            response.raise_for_status()  # Raise on HTTP errors
            
            # Parse response
            data = response.json()
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.successful_queries += 1
            self.total_latency_ms += latency_ms
            
            # Extract answer and metadata
            answer = data.get('content', '').strip()
            tokens_predicted = data.get('tokens_predicted', 0)
            self.total_tokens_generated += tokens_predicted
            
            if not answer:
                logger.warning("BitNet returned empty response")
                answer = "[BitNet generated no response]"
            
            return InferenceResponse(
                answer=answer,
                confidence=0.75,  # Fixed confidence for MVP (tune in Phase 4)
                engine_name=self.engine_name,
                sources=["bitnet"],
                latency_ms=latency_ms,
                metadata={
                    'tokens_predicted': tokens_predicted,
                    'model': data.get('model', 'unknown'),
                    'stop_type': data.get('stop_type', 'unknown'),
                    'temperature': self.temperature,
                    'query_count': self.query_count,
                    'tokens_per_second': (
                        tokens_predicted / (latency_ms / 1000)
                        if latency_ms > 0 else 0
                    ),
                }
            )
        
        except requests.Timeout:
            self.failed_queries += 1
            logger.error(f"BitNet request timeout after {self.timeout_seconds}s")
            raise RuntimeError(
                f"BitNet server timeout (>{self.timeout_seconds}s) - "
                f"query too complex or server overloaded"
            )
        
        except requests.ConnectionError as e:
            self.failed_queries += 1
            logger.error(f"Could not connect to BitNet server: {e}")
            raise RuntimeError(
                f"Cannot connect to BitNet at {self.server_url} - "
                f"make sure llama-server.exe is running"
            )
        
        except requests.HTTPError as e:
            self.failed_queries += 1
            logger.error(f"BitNet HTTP error: {e}")
            raise RuntimeError(f"BitNet server error: {e}")
        
        except ValueError as e:
            self.failed_queries += 1
            logger.error(f"Could not parse BitNet response: {e}")
            raise RuntimeError(f"Invalid BitNet response: {e}")
        
        except Exception as e:
            self.failed_queries += 1
            logger.error(f"BitNet query failed: {e}")
            raise RuntimeError(f"BitNet query failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Return BitNet engine statistics."""
        avg_latency = (
            self.total_latency_ms / self.successful_queries
            if self.successful_queries > 0
            else 0.0
        )
        
        avg_tokens = (
            self.total_tokens_generated / self.successful_queries
            if self.successful_queries > 0
            else 0.0
        )
        
        success_rate = (
            (self.successful_queries / self.query_count) * 100
            if self.query_count > 0
            else 0.0
        )
        
        return {
            'query_count': self.query_count,
            'successful_queries': self.successful_queries,
            'failed_queries': self.failed_queries,
            'success_rate_percent': success_rate,
            'avg_latency_ms': avg_latency,
            'total_latency_ms': self.total_latency_ms,
            'total_tokens_generated': self.total_tokens_generated,
            'avg_tokens_per_query': avg_tokens,
            'server_url': self.server_url,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
        }
    
    def shutdown(self) -> None:
        """Clean up resources (doesn't shut down server)."""
        # BitNet server is external process, we don't control its lifecycle
        logger.info("BitNetEngine shut down (note: server process still running)")


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Example showing how the BitNet engine works.
    
    Note: This requires BitNet server running:
        llama-server.exe --model <model.gguf> -ngl 20 -c 512
    """
    
    print("="*70)
    print("BITNET ENGINE - EXAMPLE USAGE")
    print("="*70)
    
    # Initialize BitNet engine
    try:
        engine = BitNetEngine(
            server_url="http://127.0.0.1:8080",
            max_tokens=100,
            temperature=0.7
        )
        print(f"\n✓ Initialized BitNetEngine")
        print(f"  Engine name: {engine.engine_name}")
        print(f"  Server URL: {engine.server_url}")
        print(f"  Available: {engine.is_available()}")
        print(f"  Max tokens: {engine.max_tokens}")
        print(f"  Temperature: {engine.temperature}")
        
    except ValueError as e:
        print(f"\n✗ Failed to initialize: {e}")
        exit(1)
    
    # Check if server is running
    if not engine.is_available():
        print("\n⚠ WARNING: BitNet server not reachable!")
        print("  Start the server with:")
        print("    llama-server.exe --model <model.gguf> -ngl 20 -c 512")
        print("\nContinuing with demonstration...")
    
    # Test queries
    test_queries = [
        "What is photosynthesis?",
        "Explain the water cycle",
    ]
    
    print("\n" + "="*70)
    print("BITNET QUERY EXAMPLES")
    print("="*70)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"  Sending to BitNet server...")
        
        request = InferenceRequest(
            user_query=query,
            max_tokens=100,
            temperature=0.7
        )
        
        try:
            response = engine.query(request)
            
            print(f"  Status: SUCCESS")
            print(f"  Confidence: {response.confidence:.4f}")
            print(f"  Latency: {response.latency_ms:.2f}ms")
            print(f"  Tokens: {response.metadata.get('tokens_predicted', 0)}")
            print(f"  Tokens/sec: {response.metadata.get('tokens_per_second', 0):.1f}")
            print(f"  Answer: {response.answer[:100]}...")
        
        except RuntimeError as e:
            print(f"  Status: ERROR")
            print(f"  Error: {e}")
        
        except Exception as e:
            print(f"  Status: UNEXPECTED ERROR")
            print(f"  Error: {e}")
    
    # Print statistics
    print("\n" + "="*70)
    print("BITNET ENGINE STATISTICS")
    print("="*70)
    
    stats = engine.get_stats()
    print(f"Queries processed: {stats['query_count']}")
    print(f"Successful: {stats['successful_queries']}")
    print(f"Failed: {stats['failed_queries']}")
    
    if stats['query_count'] > 0:
        print(f"\nSuccess rate: {stats['success_rate_percent']:.1f}%")
        print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"Avg tokens/query: {stats['avg_tokens_per_query']:.1f}")
        
        if stats['total_tokens_generated'] > 0:
            total_seconds = stats['total_latency_ms'] / 1000
            print(f"Overall tokens/sec: {stats['total_tokens_generated'] / total_seconds:.1f}")
    
    print("\n✓ BitNetEngine example complete")
    
    # Show server startup instructions
    print("\n" + "="*70)
    print("TO RUN WITH BITNET SERVER:")
    print("="*70)
    print("""
1. Start BitNet server in separate terminal:
   cd B:\\ai\\llm\\kitbash\\bitnet\\build\\bin
   llama-server.exe --model ..\\..\\models\\BitNet-b1_58-3B\\<model>.gguf -ngl 20 -c 512

2. Run this script

3. You should see BitNet responses instead of connection errors
""")
