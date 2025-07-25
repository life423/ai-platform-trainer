#!/usr/bin/env python3
"""
Script to identify and help clean up old/deprecated code in the AI Platform Trainer project.
Run this to get a report of what needs to be cleaned up.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def find_deprecated_code(root_path: str) -> List[Tuple[str, int, str]]:
    """Find all deprecated markers in the codebase."""
    deprecated_patterns = [
        r'DEPRECATED',
        r'deprecated',
        r'TODO.*remove',
        r'legacy',
        r'OLD CODE',
        r'obsolete',
        r'# HACK',
        r'# XXX',
        r'placeholder',
        r'temporary'
    ]
    
    results = []
    
    for root, dirs, files in os.walk(root_path):
        # Skip hidden directories and venv
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv']
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            for pattern in deprecated_patterns:
                                if re.search(pattern, line, re.IGNORECASE):
                                    results.append((file_path, line_num, line.strip()))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return results

def find_duplicate_files(root_path: str) -> dict:
    """Find potentially duplicate files based on naming patterns."""
    file_groups = {}
    
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv']
        
        for file in files:
            if file.endswith('.py'):
                # Extract base name (e.g., 'enemy' from 'enemy_play.py')
                base_name = file.replace('.py', '').split('_')[0]
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append(os.path.join(root, file))
    
    # Filter to only show groups with multiple files
    duplicates = {k: v for k, v in file_groups.items() if len(v) > 1}
    return duplicates

def find_broken_imports(root_path: str) -> List[Tuple[str, int, str]]:
    """Find imports that reference non-existent modules."""
    broken_imports = []
    
    # Known broken imports
    known_broken = ['unified_launcher', 'engine.core.service_locator']
    
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv']
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            for broken in known_broken:
                                if broken in line and ('import' in line or 'from' in line):
                                    broken_imports.append((file_path, line_num, line.strip()))
                except Exception:
                    pass
    
    return broken_imports

def main():
    """Run the cleanup analysis."""
    root_path = os.path.dirname(os.path.abspath(__file__))
    print(f"Analyzing codebase at: {root_path}\n")
    
    # Find deprecated code
    print("=" * 80)
    print("DEPRECATED CODE FOUND:")
    print("=" * 80)
    deprecated = find_deprecated_code(root_path)
    for file_path, line_num, line in deprecated[:20]:  # Show first 20
        rel_path = os.path.relpath(file_path, root_path)
        print(f"{rel_path}:{line_num} - {line}")
    if len(deprecated) > 20:
        print(f"\n... and {len(deprecated) - 20} more instances")
    
    # Find duplicate files
    print("\n" + "=" * 80)
    print("POTENTIAL DUPLICATE FILES:")
    print("=" * 80)
    duplicates = find_duplicate_files(root_path)
    for base_name, files in duplicates.items():
        if len(files) > 2:  # Only show significant duplicates
            print(f"\n{base_name}:")
            for file in files:
                rel_path = os.path.relpath(file, root_path)
                print(f"  - {rel_path}")
    
    # Find broken imports
    print("\n" + "=" * 80)
    print("BROKEN IMPORTS:")
    print("=" * 80)
    broken = find_broken_imports(root_path)
    for file_path, line_num, line in broken:
        rel_path = os.path.relpath(file_path, root_path)
        print(f"{rel_path}:{line_num} - {line}")
    
    print("\n" + "=" * 80)
    print("CLEANUP RECOMMENDATIONS:")
    print("=" * 80)
    print("1. Fix broken imports in main.py and __main__.py")
    print("2. Remove deprecated service_locator.py")
    print("3. Consolidate entity implementations (enemy_*.py, player_*.py)")
    print("4. Remove duplicate state machine implementations")
    print("5. Clean up placeholder and commented code")
    print("6. Remove deprecated missile_ai_controller.py")
    
    print(f"\nTotal deprecated markers found: {len(deprecated)}")
    print(f"Total files with potential duplicates: {sum(len(f) for f in duplicates.values())}")
    print(f"Total broken imports: {len(broken)}")

if __name__ == "__main__":
    main()