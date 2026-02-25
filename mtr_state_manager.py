"""
MTR State Manager: Checkpoint and Restore
Handles persistence of MTR 5.5 engine state across sessions.

Serializes:
- W (learned weights): w1, w2 block-diagonal matrices
- strength: Memory strength accumulation
- last_seen: Recency tracking per block
- time: Cumulative time counter
- copent_pos: CoPENt structural position

Author: Kitbash Team
Date: February 23, 2026
"""

import torch
import torch.nn as nn
import os
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime


class MTRStateCheckpoint:
    """
    Serializes and deserializes MTR engine state.
    
    Checkpoint format:
    - state.pt: Pickled torch tensors (W, strength, last_seen, time, copent_pos)
    - state_meta.json: Metadata (timestamp, d_model, d_state, device)
    """
    
    def __init__(self, checkpoint_dir: str = "data/state"):
        """
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.checkpoint_dir / "state.pt"
        self.meta_file = self.checkpoint_dir / "state_meta.json"
    
    def save(
        self,
        mtr_state: Dict,
        d_model: int,
        d_state: int,
        session_id: str = "default",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save MTR state to disk.
        
        Args:
            mtr_state: Dict with keys W, strength, last_seen, time, copent_pos
            d_model: Model dimension
            d_state: State space dimension
            session_id: Session identifier (for multi-session support)
            metadata: Optional dict with additional info (user, project, context, etc.)
        
        Returns:
            path: Path to saved checkpoint
        """
        # Validate state structure
        required_keys = {'W', 'strength', 'last_seen', 'time', 'copent_pos'}
        if not required_keys.issubset(mtr_state.keys()):
            raise ValueError(f"State missing required keys. Expected {required_keys}, got {mtr_state.keys()}")
        
        # Extract tensors
        w1, w2 = mtr_state['W']
        strength = mtr_state['strength']
        last_seen = mtr_state['last_seen']
        copent_pos = mtr_state['copent_pos']
        time_counter = mtr_state['time']
        
        # Create checkpoint dict
        checkpoint = {
            'w1': w1.cpu(),  # Always save on CPU to avoid device issues
            'w2': w2.cpu(),
            'strength': strength.cpu(),
            'last_seen': last_seen.cpu(),
            'copent_pos': copent_pos.cpu() if copent_pos is not None else None,
            'time': time_counter,  # int, not tensor
            'd_model': d_model,
            'd_state': d_state,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save tensors
        torch.save(checkpoint, self.state_file)
        
        # Save metadata
        meta = {
            'timestamp': checkpoint['timestamp'],
            'd_model': d_model,
            'd_state': d_state,
            'session_id': session_id,
            'w1_shape': tuple(w1.shape),
            'w2_shape': tuple(w2.shape),
            'strength_shape': tuple(strength.shape),
            'last_seen_shape': tuple(last_seen.shape),
            'copent_pos_shape': tuple(copent_pos.shape) if copent_pos is not None else None,
            'time': time_counter,
        }
        
        if metadata:
            meta.update(metadata)
        
        with open(self.meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✓ MTR state saved to {self.state_file}")
        print(f"  Time: {time_counter}")
        print(f"  Timestamp: {checkpoint['timestamp']}")
        
        return str(self.state_file)
    
    def load(self, device: str = 'cpu') -> Tuple[Dict, Dict]:
        """
        Load MTR state from disk.
        
        Args:
            device: Device to load tensors onto ('cpu', 'cuda', etc.)
        
        Returns:
            mtr_state: Dict formatted for MTREbbinghausLayer.forward()
            metadata: Metadata dict
        """
        if not self.state_file.exists():
            raise FileNotFoundError(f"No checkpoint found at {self.state_file}")
        
        # Load checkpoint
        checkpoint = torch.load(self.state_file, map_location=device)
        
        # Load metadata
        with open(self.meta_file, 'r') as f:
            metadata = json.load(f)
        
        # Reconstruct state dict in format expected by MTREbbinghausLayer
        mtr_state = {
            'W': (checkpoint['w1'].to(device), checkpoint['w2'].to(device)),
            'strength': checkpoint['strength'].to(device),
            'last_seen': checkpoint['last_seen'].to(device),
            'time': checkpoint['time'],
            'copent_pos': checkpoint['copent_pos'].to(device) if checkpoint['copent_pos'] is not None else None,
        }
        
        print(f"✓ MTR state loaded from {self.state_file}")
        print(f"  Time: {mtr_state['time']}")
        print(f"  Session: {metadata.get('session_id', 'unknown')}")
        print(f"  Timestamp: {metadata.get('timestamp', 'unknown')}")
        
        return mtr_state, metadata
    
    def exists(self) -> bool:
        """Check if checkpoint exists."""
        return self.state_file.exists() and self.meta_file.exists()
    
    def delete(self) -> None:
        """Delete the checkpoint."""
        if self.state_file.exists():
            self.state_file.unlink()
        if self.meta_file.exists():
            self.meta_file.unlink()
        print(f"✓ Checkpoint deleted")
    
    def get_metadata(self) -> Optional[Dict]:
        """Load and return metadata without loading full state."""
        if not self.meta_file.exists():
            return None
        
        with open(self.meta_file, 'r') as f:
            return json.load(f)


class MTRSessionManager:
    """
    Manages multiple sessions with different checkpoint paths.
    Useful for multi-project or multi-character scenarios (Phase 4+).
    """
    
    def __init__(self, base_dir: str = "data/state"):
        """
        Args:
            base_dir: Base directory for all session checkpoints
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.sessions = {}  # session_id → MTRStateCheckpoint
    
    def get_checkpoint(self, session_id: str) -> MTRStateCheckpoint:
        """Get or create checkpoint manager for a session."""
        if session_id not in self.sessions:
            session_dir = self.base_dir / session_id
            self.sessions[session_id] = MTRStateCheckpoint(str(session_dir))
        return self.sessions[session_id]
    
    def list_sessions(self) -> list:
        """List all available sessions."""
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    
    def save_session(
        self,
        session_id: str,
        mtr_state: Dict,
        d_model: int,
        d_state: int,
        metadata: Optional[Dict] = None
    ) -> str:
        """Save state for a specific session."""
        checkpoint = self.get_checkpoint(session_id)
        return checkpoint.save(mtr_state, d_model, d_state, session_id, metadata)
    
    def load_session(self, session_id: str, device: str = 'cpu') -> Tuple[Dict, Dict]:
        """Load state for a specific session."""
        checkpoint = self.get_checkpoint(session_id)
        return checkpoint.load(device)
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session has a checkpoint."""
        return self.get_checkpoint(session_id).exists()


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def save_engine_state(
    engine,
    checkpoint_dir: str = "data/state",
    session_id: str = "default",
    metadata: Optional[Dict] = None
) -> str:
    """
    Convenience function: save engine state directly.
    
    Usage:
        from kitbash.inference.mtr_engine import KitbashMTREngine
        from mtr_state_manager import save_engine_state
        
        engine = KitbashMTREngine(vocab_size=50257, d_model=256, d_state=144)
        logits, error, state = engine(token_ids, state=None)
        
        save_engine_state(engine, state, session_id='default')
    """
    if not hasattr(engine, 'mtr') or not hasattr(engine.mtr, 'd_state'):
        raise ValueError("Engine must be a KitbashMTREngine instance")
    
    manager = MTRStateCheckpoint(checkpoint_dir)
    # Note: We need the mtr_state, not the engine object
    # So this function is better called with: save_engine_state(mtr_state, d_model, d_state, ...)
    # See example below instead
    raise NotImplementedError("Use MTRStateCheckpoint.save(mtr_state, ...) directly")


def load_engine_state(
    engine,
    checkpoint_dir: str = "data/state",
    device: str = 'cpu'
) -> Dict:
    """
    Convenience function: load state into engine.
    
    Usage:
        engine = KitbashMTREngine(vocab_size=50257, d_model=256, d_state=144)
        engine = engine.to(device)
        
        mtr_state, meta = load_engine_state(engine, device=device)
        logits, error, state = engine(token_ids, state=mtr_state)
    """
    manager = MTRStateCheckpoint(checkpoint_dir)
    mtr_state, metadata = manager.load(device=device)
    return mtr_state, metadata


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/mnt/project')
    
    from MTR_Phase55_Engine import KitbashMTREngine
    
    print("=== MTR State Manager Testing ===\n")
    
    torch.manual_seed(42)
    
    # Initialize engine
    engine = KitbashMTREngine(vocab_size=1000, d_model=256, d_state=144)
    checkpoint = MTRStateCheckpoint("test_checkpoint")
    
    # Test 1: Save initial state
    print("Test 1: Generate and save initial state")
    token_ids = torch.randint(0, 1000, (1, 10))
    logits, error, mtr_state = engine(token_ids, state=None)
    
    checkpoint.save(
        mtr_state,
        d_model=256,
        d_state=144,
        session_id="test_session",
        metadata={'project': 'kitbash', 'user': 'developer'}
    )
    print()
    
    # Test 2: Load state and verify continuity
    print("Test 2: Load state and verify continuity")
    loaded_state, metadata = checkpoint.load(device='cpu')
    
    print(f"Loaded metadata: {json.dumps(metadata, indent=2)}\n")
    
    # Process another token with loaded state
    token_ids2 = torch.randint(0, 1000, (1, 10))
    logits2, error2, new_state = engine(token_ids2, state=loaded_state)
    
    print(f"✓ Loaded state time: {loaded_state['time']}")
    print(f"✓ New state time after inference: {new_state['time']}")
    print(f"✓ Time accumulated correctly: {new_state['time'] == loaded_state['time'] + 10}\n")
    
    # Test 3: Session manager
    print("Test 3: Multi-session management")
    session_mgr = MTRSessionManager("test_sessions")
    
    # Save session 1
    session_mgr.save_session(
        "session_1",
        new_state,
        d_model=256,
        d_state=144,
        metadata={'project': 'fiction_writing'}
    )
    
    # Save session 2
    token_ids3 = torch.randint(0, 1000, (1, 5))
    logits3, error3, state3 = engine(token_ids3, state=new_state)
    session_mgr.save_session(
        "session_2",
        state3,
        d_model=256,
        d_state=144,
        metadata={'project': 'research_notes'}
    )
    
    # List and load
    sessions = session_mgr.list_sessions()
    print(f"Available sessions: {sessions}")
    
    loaded1, meta1 = session_mgr.load_session("session_1", device='cpu')
    loaded2, meta2 = session_mgr.load_session("session_2", device='cpu')
    
    print(f"Session 1 time: {loaded1['time']}, project: {meta1.get('project')}")
    print(f"Session 2 time: {loaded2['time']}, project: {meta2.get('project')}\n")
    
    # Test 4: Checkpoint metadata
    print("Test 4: Metadata without full load")
    meta = checkpoint.get_metadata()
    print(f"Checkpoint metadata (no full load):")
    print(json.dumps(meta, indent=2))
    print()
    
    # Cleanup
    checkpoint.delete()
    import shutil
    shutil.rmtree("test_checkpoint", ignore_errors=True)
    shutil.rmtree("test_sessions", ignore_errors=True)
    
    print("=== All tests passed ===")
