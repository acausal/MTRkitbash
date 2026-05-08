"""
Grain Crystallizer - Persist Grains to Storage

Persists crushed ternary grains to cartridge storage.

Responsibilities:
- Save grain JSON files to cartridges/{name}/grains/
- Update cartridge manifest with grain_inventory
- Track crystallization metadata
- Enable grain loading for activation layer

Author: Kitbash Team
Date: February 2026
Phase: 2C Consolidation → Modularized (May 2026)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone


class GrainCrystallizer:
    """
    Persists crushed ternary grains to cartridge storage.
    
    Responsibilities:
    - Save grain JSON files to cartridges/{name}/grains/
    - Update cartridge manifest with grain_inventory
    - Track crystallization metadata
    - Enable grain loading for activation layer
    
    From grain_crystallizer.py
    """
    
    def __init__(self, cartridges_dir: str = "./cartridges"):
        """Initialize crystallizer."""
        self.cartridges_dir = Path(cartridges_dir)
    
    def crystallize_grains(self, crushed_grains: List[Dict[str, Any]],
                          cartridge_id: str) -> Dict[str, Any]:
        """
        Save crushed grains for a specific cartridge.
        
        Args:
            crushed_grains: List of ternary grain dicts from TernaryCrush
            cartridge_id: Name of cartridge these grains belong to
        
        Returns:
            Crystallization result with file paths and statistics
        """
        
        cartridge_path = self.cartridges_dir / f"{cartridge_id}.kbc"
        grains_dir = cartridge_path / "grains"
        
        grains_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            'cartridge_id': cartridge_id,
            'grain_count': 0,
            'grain_files': [],
            'manifest_updated': False,
            'errors': [],
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        
        # Save each grain
        for grain in crushed_grains:
            try:
                grain_file = self._save_grain_file(grain, grains_dir)
                result['grain_files'].append(grain_file)
                result['grain_count'] += 1
            except Exception as e:
                result['errors'].append(f"Grain {grain.get('grain_id')}: {str(e)}")
        
        # Update manifest
        try:
            self._update_manifest(grains_dir.parent, crushed_grains, result)
            result['manifest_updated'] = True
        except Exception as e:
            result['errors'].append(f"Manifest update: {str(e)}")
        
        return result
    
    def _save_grain_file(self, grain: Dict[str, Any], grains_dir: Path) -> str:
        """Save individual grain to JSON file."""
        
        grain_id = grain.get('grain_id')
        if not grain_id:
            raise ValueError("Grain missing grain_id")
        
        grain_json = {
            'grain_id': grain['grain_id'],
            'fact_id': grain['fact_ids'][0] if grain['fact_ids'] else None,
            'fact_ids': grain['fact_ids'],
            'cartridge_source': grain['cartridge_id'],
            'axiom_link': 'domain_concept',
            'lock_state': 'Sicherman_Validated',
            'weight': grain['weight'],
            'delta': grain['delta'],
            'confidence': grain['confidence'],
            'cycles_locked': grain['cycles_locked'],
            'validation_timestamp': datetime.now(timezone.utc).isoformat(),
            'pointer_map': grain['pointer_map'],
        }
        
        grain_file = grains_dir / f"{grain_id}.json"
        with open(grain_file, 'w') as f:
            json.dump(grain_json, f, indent=2)
        
        return str(grain_file.relative_to(self.cartridges_dir))
    
    def _update_manifest(self, cartridge_path: Path, 
                        crushed_grains: List[Dict], 
                        result: Dict) -> None:
        """Update cartridge manifest with grain inventory."""
        
        manifest_file = cartridge_path / "manifest.json"
        
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {
                'cartridge_id': cartridge_path.name.replace('.kbc', ''),
                'created': datetime.now(timezone.utc).isoformat(),
            }
        
        if 'grain_inventory' not in manifest:
            manifest['grain_inventory'] = {}
        
        for grain in crushed_grains:
            grain_id = grain['grain_id']
            manifest['grain_inventory'][grain_id] = {
                'fact_ids': grain['fact_ids'],
                'axiom_link': 'domain_concept',
                'lock_state': 'Sicherman_Validated',
                'confidence': grain['confidence'],
                'weight': grain['weight'],
                'crystallization_timestamp': datetime.now(timezone.utc).isoformat(),
            }
        
        manifest['grain_crystallization'] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'grain_count': len(manifest['grain_inventory']),
            'phase': '2C_consolidated',
        }
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def load_grain(self, cartridge_id: str, grain_id: str) -> Optional[Dict[str, Any]]:
        """Load a crystallized grain from disk."""
        
        grain_file = self.cartridges_dir / f"{cartridge_id}.kbc" / "grains" / f"{grain_id}.json"
        
        if not grain_file.exists():
            return None
        
        with open(grain_file, 'r') as f:
            return json.load(f)
    
    def load_all_grains(self, cartridge_id: str) -> Dict[str, Dict[str, Any]]:
        """Load all crystallized grains for a cartridge."""
        
        grains_dir = self.cartridges_dir / f"{cartridge_id}.kbc" / "grains"
        
        if not grains_dir.exists():
            return {}
        
        grains = {}
        for grain_file in grains_dir.glob("*.json"):
            try:
                with open(grain_file, 'r') as f:
                    grain = json.load(f)
                    grain_id = grain.get('grain_id')
                    if grain_id:
                        grains[grain_id] = grain
            except Exception as e:
                print(f"Warning: Could not load grain {grain_file}: {e}")
        
        return grains
    
    def get_manifests(self) -> Dict[str, Dict[str, Any]]:
        """Load all cartridge manifests with grain inventory."""
        
        manifests = {}
        
        for cartridge_dir in self.cartridges_dir.glob("*.kbc"):
            manifest_file = cartridge_dir / "manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, 'r') as f:
                        manifest = json.load(f)
                        cartridge_id = cartridge_dir.name.replace('.kbc', '')
                        manifests[cartridge_id] = manifest
                except Exception as e:
                    print(f"Warning: Could not load manifest {manifest_file}: {e}")
        
        return manifests


class GrainCrystallizationReport:
    """Generate and save crystallization summary report."""
    
    def __init__(self, cartridges_dir: str = "./cartridges"):
        self.cartridges_dir = Path(cartridges_dir)
        self.report = {
            'phase': '2C_consolidated',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cartridges': {},
            'summary': {
                'total_grains': 0,
                'total_cartridges': 0,
                'total_size': 0,
            }
        }
    
    def add_crystallization_result(self, result: Dict[str, Any]) -> None:
        """Add a crystallization result to the report."""
        
        cartridge_id = result['cartridge_id']
        self.report['cartridges'][cartridge_id] = {
            'grain_count': result['grain_count'],
            'grain_files': result['grain_files'],
            'manifest_updated': result['manifest_updated'],
            'errors': result['errors'],
            'timestamp': result['timestamp'],
        }
        
        self.report['summary']['total_grains'] += result['grain_count']
        self.report['summary']['total_cartridges'] += 1
    
    def calculate_sizes(self) -> None:
        """Calculate total crystallized grain size."""
        
        total_size = 0
        
        for cartridge_id in self.report['cartridges']:
            grain_files = self.report['cartridges'][cartridge_id]['grain_files']
            for grain_file in grain_files:
                filepath = self.cartridges_dir / grain_file
                if filepath.exists():
                    total_size += filepath.stat().st_size
        
        self.report['summary']['total_size'] = total_size
    
    def save(self, output_file: str = "./phase2c_crystallization_report.json") -> None:
        """Save report to JSON file."""
        
        self.calculate_sizes()
        
        with open(output_file, 'w') as f:
            json.dump(self.report, f, indent=2)
    
    def print_summary(self) -> None:
        """Print report summary."""
        
        self.calculate_sizes()
        
        print("\n" + "="*70)
        print("GRAIN CRYSTALLIZATION REPORT")
        print("="*70)
        print(f"Timestamp: {self.report['timestamp']}")
        print(f"Cartridges processed: {self.report['summary']['total_cartridges']}")
        print(f"Total grains crystallized: {self.report['summary']['total_grains']}")
        print(f"Total grain storage: {self.report['summary']['total_size']:,} bytes")
        
        print("\nPer-cartridge breakdown:")
        for cart_id, cart_report in self.report['cartridges'].items():
            print(f"  {cart_id}:")
            print(f"    - Grains: {cart_report['grain_count']}")
            print(f"    - Manifest updated: {cart_report['manifest_updated']}")
            if cart_report['errors']:
                print(f"    - Errors: {len(cart_report['errors'])}")
        
        print("="*70 + "\n")
