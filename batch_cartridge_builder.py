#!/usr/bin/env python3
"""
Batch Cartridge Builder
=======================

Process multiple seed files in bulk, converting them all to {domain}_general.kbc cartridges.

Usage:
    python batch_cartridge_builder.py --seed-dir ./seeds/ --output ./cartridges/

Or with custom domain extraction:
    python batch_cartridge_builder.py --seed-dir ./seeds/ --output ./cartridges/ --naming-pattern "{base}"

Expects seed files in format:
    domain_seed.md
    domain_seed.txt
    domain_seed.json
    etc.

Extracts domain name by removing "_seed" suffix.
Examples:
    physics_seed.md → physics_general.kbc
    formal_logic_seed.txt → formal_logic_general.kbc
    biochemistry_seed.json → biochemistry_general.kbc
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import sys
import re

# Add parent directory to path to import cartridge_builder
sys.path.insert(0, str(Path(__file__).parent))
from cartridge_builder import CartridgeBuilder


class BatchCartridgeBuilder:
    """Process multiple seed files in batch"""
    
    def __init__(self, seed_dir: str, output_dir: str = "./cartridges"):
        self.seed_dir = Path(seed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.seed_dir.exists():
            raise FileNotFoundError(f"Seed directory not found: {seed_dir}")
        
        self.results = {
            'successful': [],
            'skipped': [],
            'failed': []
        }
    
    def discover_seeds(self) -> List[Path]:
        """Find all seed files in the directory"""
        # Look for files with "seed" in the name (case-insensitive)
        # Matches: _seed, Seed, SEED, etc.
        seed_files = []
        
        for file_path in self.seed_dir.iterdir():
            if file_path.is_file() and "seed" in file_path.name.lower():
                seed_files.append(file_path)
        
        return sorted(seed_files)
    
    def extract_domain(self, filename: str) -> str:
        """
        Extract domain name from seed filename.
        
        Examples:
            physics_seed.md → physics
            formal_logic_seed.txt → formal_logic
            PhysicsSeed.md → physics
            FormalLogicSeed.txt → formal_logic
            BiochemistrySeed.md → biochemistry
        """
        # Remove extension
        name_without_ext = filename.rsplit('.', 1)[0]
        
        # Remove seed suffix (case-insensitive)
        if name_without_ext.lower().endswith("seed"):
            domain = name_without_ext[:-4]  # Remove "Seed" or "seed"
        else:
            domain = name_without_ext
        
        # Convert CamelCase to snake_case
        # Insert underscore before uppercase letters (except first)
        domain = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', domain)
        domain = re.sub('([a-z0-9])([A-Z])', r'\1_\2', domain)
        
        # Convert to lowercase
        domain = domain.lower()
        
        return domain
    
    def build_from_seed(self, seed_file: Path, faction: str = "general") -> bool:
        """
        Build a single cartridge from a seed file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            domain = self.extract_domain(seed_file.name)
            
            print(f"\n{'='*70}")
            print(f"Processing: {seed_file.name}")
            print(f"Domain: {domain}, Faction: {faction}")
            print(f"{'='*70}")
            
            # Detect format from extension
            format_map = {
                '.md': 'markdown',
                '.txt': 'text',
                '.csv': 'csv',
                '.json': 'json'
            }
            file_format = format_map.get(seed_file.suffix.lower(), 'text')
            
            # Create builder
            builder = CartridgeBuilder(domain, faction=faction, output_dir=str(self.output_dir))
            
            # Load from seed file
            if file_format == "markdown":
                builder.load_from_markdown(str(seed_file), interactive=False)
            elif file_format == "csv":
                builder.load_from_csv(str(seed_file))
            elif file_format == "json":
                builder.load_from_json(str(seed_file))
            else:
                builder.load_from_text(str(seed_file))
            
            # Preview
            builder.preview()
            
            # Save with seed file reference
            builder.save(seed_file=str(seed_file), force=True)
            
            self.results['successful'].append({
                'domain': domain,
                'seed_file': seed_file.name,
                'faction': faction,
                'fact_count': len(builder.facts)
            })
            
            return True
        
        except Exception as e:
            print(f"✗ Error processing {seed_file.name}: {e}")
            self.results['failed'].append({
                'seed_file': seed_file.name,
                'error': str(e)
            })
            return False
    
    def build_all(self, faction: str = "general") -> None:
        """Build all discovered seed files"""
        seed_files = self.discover_seeds()
        
        if not seed_files:
            print(f"No seed files found in {self.seed_dir}")
            return
        
        print(f"\n{'='*70}")
        print(f"BATCH CARTRIDGE BUILDER")
        print(f"{'='*70}")
        print(f"Found {len(seed_files)} seed files")
        print(f"Output directory: {self.output_dir}")
        print(f"Faction: {faction}")
        print(f"{'='*70}\n")
        
        for i, seed_file in enumerate(seed_files, 1):
            print(f"\n[{i}/{len(seed_files)}] Processing...")
            self.build_from_seed(seed_file, faction=faction)
        
        # Print summary
        self._print_summary(seed_files)
    
    def _print_summary(self, total_files: List[Path]) -> None:
        """Print batch processing summary"""
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total files: {len(total_files)}")
        print(f"Successful: {len(self.results['successful'])}")
        print(f"Failed: {len(self.results['failed'])}")
        print(f"Skipped: {len(self.results['skipped'])}")
        
        if self.results['successful']:
            print(f"\n✓ Successful builds:")
            total_facts = 0
            for result in self.results['successful']:
                print(f"  - {result['domain']:30} ({result['fact_count']} facts)")
                total_facts += result['fact_count']
            print(f"  Total facts: {total_facts}")
        
        if self.results['failed']:
            print(f"\n✗ Failed builds:")
            for result in self.results['failed']:
                print(f"  - {result['seed_file']}: {result['error']}")
        
        print(f"\nCartridges saved to: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def save_manifest(self) -> None:
        """Save a manifest of all built cartridges"""
        manifest = {
            'batch_build_info': {
                'seed_directory': str(self.seed_dir),
                'output_directory': str(self.output_dir),
                'total_cartridges': len(self.results['successful'])
            },
            'cartridges': self.results['successful'],
            'failed': self.results['failed']
        }
        
        manifest_path = self.output_dir / "BUILD_MANIFEST.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✓ Saved build manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process multiple seed files into cartridges"
    )
    parser.add_argument("--seed-dir", required=True, help="Directory containing seed files")
    parser.add_argument("--output", default="./cartridges", help="Output directory for cartridges")
    parser.add_argument("--faction", default="general", choices=["general", "fiction", "experiment"],
                        help="Faction for all cartridges (default: general)")
    parser.add_argument("--manifest", action="store_true", help="Save build manifest")
    
    args = parser.parse_args()
    
    try:
        builder = BatchCartridgeBuilder(args.seed_dir, args.output)
        builder.build_all(faction=args.faction)
        
        if args.manifest:
            builder.save_manifest()
    
    except KeyboardInterrupt:
        print("\n\n✗ Batch build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
