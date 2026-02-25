"""
MTR Learning Diagnostics
Provides inspection tools for understanding what the MTR engine is learning.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from collections import defaultdict


class MTRDiagnostics:
    """
    Tracks and reports on MTR learning behavior.
    """
    
    def __init__(self):
        self.state_snapshots = []  # List of (step, state_dict) tuples
        self.error_history = []     # (step, mean_error, max_error)
        self.epistemic_history = defaultdict(list)  # layer_name → [(step, mean_salience), ...]
        self.dissonance_log = []    # (step, dissonance_active_count, total_tokens)
        self.step = 0
    
    def log_inference(self, 
                     error_signal: torch.Tensor,
                     epistemic_snapshot: Dict,
                     mtr_state: Dict,
                     dissonance_result: Optional[Dict] = None):
        """
        Log a single inference step.
        
        Args:
            error_signal: (batch, seq_len, 1) from MTR
            epistemic_snapshot: Dict from epistemic_router
            mtr_state: Current MTR state dict
            dissonance_result: Optional dissonance sensor output
        """
        self.step += 1
        
        # Error signal stats
        mean_error = float(error_signal.mean())
        max_error = float(error_signal.max())
        self.error_history.append((self.step, mean_error, max_error))
        
        # Epistemic layer salience
        for layer_name, (representation, salience) in epistemic_snapshot.items():
            mean_salience = float(salience.mean())
            self.epistemic_history[layer_name].append((self.step, mean_salience))
        
        # Dissonance tracking
        if dissonance_result:
            active_count = int(dissonance_result['dissonance_active'].sum())
            total_tokens = dissonance_result['dissonance_active'].numel()
            self.dissonance_log.append((self.step, active_count, total_tokens))
        
        # Snapshot state for comparison
        self.state_snapshots.append((
            self.step,
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
             for k, v in mtr_state.items()}
        ))
    
    def get_weight_changes(self) -> Dict[str, float]:
        """
        Compare first and last state snapshots to show weight movement.
        """
        if len(self.state_snapshots) < 2:
            return {"note": "Insufficient snapshots for comparison"}
        
        first_step, first_state = self.state_snapshots[0]
        last_step, last_state = self.state_snapshots[-1]
        
        changes = {}
        
        # W1 and W2 changes
        if 'W' in first_state and 'W' in last_state:
            w1_first, w2_first = first_state['W']
            w1_last, w2_last = last_state['W']
            
            w1_change = float(torch.norm(w1_last - w1_first))
            w2_change = float(torch.norm(w2_last - w2_first))
            
            changes['W1_norm_change'] = w1_change
            changes['W2_norm_change'] = w2_change
        
        # Strength changes
        if 'strength' in first_state and 'strength' in last_state:
            strength_first = first_state['strength']
            strength_last = last_state['strength']
            
            strength_change = float(torch.norm(strength_last - strength_first))
            changes['strength_norm_change'] = strength_change
        
        return changes
    
    def report(self) -> str:
        """
        Generate a human-readable diagnostics report.
        """
        lines = []
        lines.append("\n" + "="*70)
        lines.append("MTR LEARNING DIAGNOSTICS")
        lines.append("="*70)
        
        # Error signal trend
        if self.error_history:
            lines.append("\nError Signal Trend:")
            errors = [e[1] for e in self.error_history]
            first_error = errors[0]
            last_error = errors[-1]
            avg_error = sum(errors) / len(errors)
            
            trend = "↑ increasing" if last_error > first_error else "↓ decreasing"
            lines.append(f"  First: {first_error:.4f}")
            lines.append(f"  Last:  {last_error:.4f} {trend}")
            lines.append(f"  Avg:   {avg_error:.4f}")
            lines.append(f"  Max:   {max(errors):.4f}")
        
        # Epistemic layer activity
        if self.epistemic_history:
            lines.append("\nEpistemic Layer Salience (final step):")
            for layer_name in sorted(self.epistemic_history.keys()):
                history = self.epistemic_history[layer_name]
                if history:
                    final_salience = history[-1][1]
                    lines.append(f"  {layer_name:15s}: {final_salience:.4f}")
        
        # Dissonance activity
        if self.dissonance_log:
            lines.append("\nDissonance Activity:")
            total_dissonances = sum(d[1] for d in self.dissonance_log)
            total_tokens = sum(d[2] for d in self.dissonance_log)
            dissonance_rate = total_dissonances / total_tokens if total_tokens > 0 else 0
            
            lines.append(f"  Total dissonant tokens: {total_dissonances} / {total_tokens}")
            lines.append(f"  Dissonance rate: {dissonance_rate:.2%}")
        
        # Weight changes
        weight_changes = self.get_weight_changes()
        if weight_changes and "note" not in weight_changes:
            lines.append("\nWeight Movement (L2 norm of change):")
            for key, val in sorted(weight_changes.items()):
                lines.append(f"  {key}: {val:.6f}")
        
        # Time tracking
        if self.state_snapshots:
            first_time = self.state_snapshots[0][1].get('time', 0)
            last_time = self.state_snapshots[-1][1].get('time', 0)
            lines.append(f"\nMTR Time Counter: {first_time} → {last_time} ({last_time - first_time} steps)")
        
        lines.append("="*70)
        return "\n".join(lines)
    
    def layer_activity_summary(self) -> str:
        """Quick summary of which layers are most active."""
        if not self.epistemic_history:
            return "No epistemic data collected"
        
        # Get final salience for each layer
        final_saliences = {}
        for layer_name, history in self.epistemic_history.items():
            if history:
                final_saliences[layer_name] = history[-1][1]
        
        # Sort by salience
        sorted_layers = sorted(final_saliences.items(), key=lambda x: x[1], reverse=True)
        
        lines = ["Epistemic Layer Activity (sorted by final salience):"]
        for layer_name, salience in sorted_layers:
            bar = "█" * int(salience * 20)  # Visual bar
            lines.append(f"  {layer_name:15s} {salience:.4f} {bar}")
        
        return "\n".join(lines)
