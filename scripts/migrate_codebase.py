#!/usr/bin/env python
"""
Migration script for AI Platform Trainer refactoring.

This script helps automate the process of migrating the codebase from the old structure
to the new enterprise-level structure. It can:
1. Create the necessary directory structure
2. Move files from old locations to new locations
3. Update import statements to match the new structure
4. Generate __init__.py files

Usage:
    python migrate_codebase.py [--dry-run] [--verbose]
"""
import argparse
import fnmatch
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# Mapping from old directory to new directory
# 'old_path': 'new_path'
FILE_MAPPING = {
    "ai_platform_trainer/ai_model/enemy_rl_agent.py": "src/ai_platform_trainer/ml/rl/enemy_agent.py",
    "ai_platform_trainer/ai_model/missile_dataset.py": "src/ai_platform_trainer/ml/training/missile_dataset.py",
    "ai_platform_trainer/ai_model/simple_missile_model.py": "src/ai_platform_trainer/ml/models/missile_model.py",
    "ai_platform_trainer/ai_model/train_enemy_rl.py": "src/ai_platform_trainer/ml/rl/train_enemy_rl.py",
    "ai_platform_trainer/ai_model/train_missile_model.py": "src/ai_platform_trainer/ml/training/train_missile_model.py",
    "ai_platform_trainer/ai_model/training_monitor.py": "src/ai_platform_trainer/ml/training/training_monitor.py",
    "ai_platform_trainer/ai_model/model_definition/enemy_movement_model.py": "src/ai_platform_trainer/ml/models/enemy_movement_model.py",
    
    "ai_platform_trainer/core/adapter.py": "src/ai_platform_trainer/core/adapter.py",
    "ai_platform_trainer/core/config_manager.py": "src/ai_platform_trainer/core/config_manager.py",
    "ai_platform_trainer/core/data_logger.py": "src/ai_platform_trainer/core/data_logger.py",
    "ai_platform_trainer/core/interfaces.py": "src/ai_platform_trainer/core/interfaces.py",
    "ai_platform_trainer/core/launcher.py": "src/ai_platform_trainer/core/launcher.py",
    "ai_platform_trainer/core/launcher_di.py": "src/ai_platform_trainer/core/launcher_di.py",
    "ai_platform_trainer/core/launcher_refactored.py": "src/ai_platform_trainer/core/launcher_refactored.py",
    "ai_platform_trainer/core/logging_config.py": "src/ai_platform_trainer/core/logging_config.py",
    "ai_platform_trainer/core/service_locator.py": "src/ai_platform_trainer/core/service_locator.py",
    
    "ai_platform_trainer/gameplay/collisions.py": "src/ai_platform_trainer/physics/collisions.py",
    "ai_platform_trainer/gameplay/common_utils.py": "src/ai_platform_trainer/utils/gameplay_utils.py",
    "ai_platform_trainer/gameplay/config.py": "src/ai_platform_trainer/core/config.py",
    "ai_platform_trainer/gameplay/display_manager.py": "src/ai_platform_trainer/rendering/display_manager.py",
    "ai_platform_trainer/gameplay/game.py": "src/ai_platform_trainer/core/game.py",
    "ai_platform_trainer/gameplay/game_di.py": "src/ai_platform_trainer/core/game_di.py",
    "ai_platform_trainer/gameplay/game_refactored.py": "src/ai_platform_trainer/core/game_refactored.py",
    "ai_platform_trainer/gameplay/input_handler.py": "src/ai_platform_trainer/core/input_handler.py",
    "ai_platform_trainer/gameplay/menu.py": "src/ai_platform_trainer/rendering/menu.py",
    "ai_platform_trainer/gameplay/missile_ai_controller.py": "src/ai_platform_trainer/ml/inference/missile_ai_controller.py",
    "ai_platform_trainer/gameplay/post_training_processor.py": "src/ai_platform_trainer/ml/training/post_training_processor.py",
    "ai_platform_trainer/gameplay/renderer.py": "src/ai_platform_trainer/rendering/renderer.py",
    "ai_platform_trainer/gameplay/renderer_di.py": "src/ai_platform_trainer/rendering/renderer_di.py",
    "ai_platform_trainer/gameplay/spawn_utils.py": "src/ai_platform_trainer/gameplay/spawn_utils.py",
    "ai_platform_trainer/gameplay/spawner.py": "src/ai_platform_trainer/gameplay/spawner.py",
    "ai_platform_trainer/gameplay/state_machine.py": "src/ai_platform_trainer/gameplay/state_machine.py",
    
    "ai_platform_trainer/entities/enemy.py": "src/ai_platform_trainer/entities/enemy.py",
    "ai_platform_trainer/entities/enemy_play.py": "src/ai_platform_trainer/entities/enemy_play.py",
    "ai_platform_trainer/entities/enemy_training.py": "src/ai_platform_trainer/entities/enemy_training.py",
    "ai_platform_trainer/entities/entity_factory.py": "src/ai_platform_trainer/entities/entity_factory.py",
    "ai_platform_trainer/entities/missile.py": "src/ai_platform_trainer/entities/missile.py",
    "ai_platform_trainer/entities/player.py": "src/ai_platform_trainer/entities/player.py",
    "ai_platform_trainer/entities/player_play.py": "src/ai_platform_trainer/entities/player_play.py",
    "ai_platform_trainer/entities/player_training.py": "src/ai_platform_trainer/entities/player_training.py",
    
    "ai_platform_trainer/utils/common_utils.py": "src/ai_platform_trainer/utils/common_utils.py",
    "ai_platform_trainer/utils/data_validator_and_trainer.py": "src/ai_platform_trainer/ml/training/data_validator.py",
    "ai_platform_trainer/utils/helpers.py": "src/ai_platform_trainer/utils/helpers.py",
    "ai_platform_trainer/utils/sprite_manager.py": "src/ai_platform_trainer/rendering/sprite_manager.py",
    
    "ai_platform_trainer/main.py": "src/ai_platform_trainer/main.py",
    
    # Add more mappings as needed
}

# Import pattern to match import statements
IMPORT_PATTERN = re.compile(r"from (ai_platform_trainer\.\w+(\.\w+)*) import (.+)|import (ai_platform_trainer\.\w+(\.\w+)*)")

# Mapping of old import paths to new import paths
IMPORT_MAPPING = {
    # AI model imports
    "ai_platform_trainer.ai_model.enemy_rl_agent": "ai_platform_trainer.ml.rl.enemy_agent",
    "ai_platform_trainer.ai_model.missile_dataset": "ai_platform_trainer.ml.training.missile_dataset",
    "ai_platform_trainer.ai_model.simple_missile_model": "ai_platform_trainer.ml.models.missile_model",
    "ai_platform_trainer.ai_model.train_enemy_rl": "ai_platform_trainer.ml.rl.train_enemy_rl",
    "ai_platform_trainer.ai_model.train_missile_model": "ai_platform_trainer.ml.training.train_missile_model",
    "ai_platform_trainer.ai_model.training_monitor": "ai_platform_trainer.ml.training.training_monitor",
    "ai_platform_trainer.ai_model.model_definition.enemy_movement_model": "ai_platform_trainer.ml.models.enemy_movement_model",
    
    # Gameplay imports
    "ai_platform_trainer.gameplay.collisions": "ai_platform_trainer.physics.collisions",
    "ai_platform_trainer.gameplay.common_utils": "ai_platform_trainer.utils.gameplay_utils",
    "ai_platform_trainer.gameplay.config": "ai_platform_trainer.core.config",
    "ai_platform_trainer.gameplay.display_manager": "ai_platform_trainer.rendering.display_manager",
    "ai_platform_trainer.gameplay.game": "ai_platform_trainer.core.game",
    "ai_platform_trainer.gameplay.game_di": "ai_platform_trainer.core.game_di",
    "ai_platform_trainer.gameplay.game_refactored": "ai_platform_trainer.core.game_refactored",
    "ai_platform_trainer.gameplay.input_handler": "ai_platform_trainer.core.input_handler",
    "ai_platform_trainer.gameplay.menu": "ai_platform_trainer.rendering.menu",
    "ai_platform_trainer.gameplay.missile_ai_controller": "ai_platform_trainer.ml.inference.missile_ai_controller",
    "ai_platform_trainer.gameplay.post_training_processor": "ai_platform_trainer.ml.training.post_training_processor",
    "ai_platform_trainer.gameplay.renderer": "ai_platform_trainer.rendering.renderer",
    "ai_platform_trainer.gameplay.renderer_di": "ai_platform_trainer.rendering.renderer_di",
    
    # Utils imports
    "ai_platform_trainer.utils.data_validator_and_trainer": "ai_platform_trainer.ml.training.data_validator",
    "ai_platform_trainer.utils.sprite_manager": "ai_platform_trainer.rendering.sprite_manager",
    
    # Add more mappings as needed
}

# Directories to create
DIRECTORIES_TO_CREATE = [
    "src/ai_platform_trainer",
    "src/ai_platform_trainer/core",
    "src/ai_platform_trainer/ml",
    "src/ai_platform_trainer/ml/models",
    "src/ai_platform_trainer/ml/training",
    "src/ai_platform_trainer/ml/rl",
    "src/ai_platform_trainer/ml/inference",
    "src/ai_platform_trainer/physics",
    "src/ai_platform_trainer/physics/cpp",
    "src/ai_platform_trainer/physics/cpp/include",
    "src/ai_platform_trainer/physics/cpp/src",
    "src/ai_platform_trainer/physics/cpp/pybind",
    "src/ai_platform_trainer/entities",
    "src/ai_platform_trainer/rendering",
    "src/ai_platform_trainer/gameplay",
    "src/ai_platform_trainer/utils",
]


def create_directory_structure(root_dir: Path, dry_run: bool = False, verbose: bool = False) -> None:
    """
    Create the necessary directory structure for the new codebase.
    
    Args:
        root_dir: The root directory of the project
        dry_run: If True, don't actually create directories
        verbose: If True, print verbose output
    """
    for directory in DIRECTORIES_TO_CREATE:
        full_path = root_dir / directory
        if not full_path.exists():
            if verbose:
                print(f"Creating directory: {full_path}")
            if not dry_run:
                full_path.mkdir(parents=True, exist_ok=True)
                # Create __init__.py file
                init_file = full_path / "__init__.py"
                if not init_file.exists():
                    if verbose:
                        print(f"Creating file: {init_file}")
                    if not dry_run:
                        init_file.touch()


def migrate_files(root_dir: Path, dry_run: bool = False, verbose: bool = False) -> None:
    """
    Move files from their old locations to new locations.
    
    Args:
        root_dir: The root directory of the project
        dry_run: If True, don't actually move files
        verbose: If True, print verbose output
    """
    for old_path, new_path in FILE_MAPPING.items():
        old_file = root_dir / old_path
        new_file = root_dir / new_path
        
        if old_file.exists():
            if verbose:
                print(f"Moving {old_file} -> {new_file}")
            
            if not dry_run:
                # Ensure the target directory exists
                new_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file to new location (instead of moving to preserve the original)
                shutil.copy2(old_file, new_file)
                
                # Update import statements in the file
                update_imports(new_file, verbose=verbose, dry_run=dry_run)
        else:
            print(f"Warning: Source file {old_file} does not exist")


def find_files_with_extension(root_dir: Path, extension: str) -> List[Path]:
    """
    Find all files with the given extension in the root_dir.
    
    Args:
        root_dir: The root directory to search in
        extension: The file extension to look for (e.g., '.py')
        
    Returns:
        List of paths to files with the given extension
    """
    return list(root_dir.glob(f"**/*{extension}"))


def update_imports(file_path: Path, verbose: bool = False, dry_run: bool = False) -> None:
    """
    Update import statements in the given file.
    
    Args:
        file_path: Path to the file to update
        verbose: If True, print verbose output
        dry_run: If True, don't actually update the file
    """
    if not file_path.exists() or not file_path.suffix == '.py':
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all import statements
        matches = IMPORT_PATTERN.finditer(content)
        modified_content = content
        replacements = []
        
        for match in matches:
            from_import = match.group(1)
            direct_import = match.group(4)
            
            if from_import and from_import in IMPORT_MAPPING:
                old_import = f"from {from_import} import {match.group(3)}"
                new_import = f"from {IMPORT_MAPPING[from_import]} import {match.group(3)}"
                replacements.append((old_import, new_import))
            elif direct_import and direct_import in IMPORT_MAPPING:
                old_import = f"import {direct_import}"
                new_import = f"import {IMPORT_MAPPING[direct_import]}"
                replacements.append((old_import, new_import))
        
        # Apply replacements
        for old_import, new_import in replacements:
            if verbose:
                print(f"  Replacing in {file_path.name}: {old_import} -> {new_import}")
            if not dry_run:
                modified_content = modified_content.replace(old_import, new_import)
        
        # Save file if changes were made
        if modified_content != content and not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main() -> None:
    """Main function to run the migration script."""
    parser = argparse.ArgumentParser(description="Migrate AI Platform Trainer codebase to new structure")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually make changes, just print what would be done")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()
    
    # Get the project root directory
    root_dir = Path.cwd()
    
    # Ensure we're in the project root
    if not (root_dir / "ai_platform_trainer").exists():
        print("Error: Script must be run from the project root directory")
        return
    
    print(f"Starting migration{'(dry run)' if args.dry_run else ''}")
    
    # Create directory structure
    create_directory_structure(root_dir, args.dry_run, args.verbose)
    
    # Move files
    migrate_files(root_dir, args.dry_run, args.verbose)
    
    # Update imports in src directory
    src_dir = root_dir / "src"
    if src_dir.exists():
        python_files = find_files_with_extension(src_dir, ".py")
        for file_path in python_files:
            if args.verbose:
                print(f"Checking imports in {file_path}")
            update_imports(file_path, args.verbose, args.dry_run)
    
    print(f"Migration {'would be ' if args.dry_run else ''}completed")
    
    if not args.dry_run:
        print("\nNext steps:")
        print("1. Run tests to ensure functionality is preserved")
        print("2. Update any imports that weren't automatically updated")
        print("3. Remove old files once migration is verified")


if __name__ == "__main__":
    main()
