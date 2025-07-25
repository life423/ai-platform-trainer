# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AI Platform Trainer
This configures the build process for creating standalone executables
"""

import sys
import os
from pathlib import Path

# Get the directory containing this spec file
spec_root = Path(SPECPATH)
project_root = spec_root

block_cipher = None

# Define the main script
main_script = project_root / 'run_game.py'

# Data files to include
data_files = [
    # Game assets
    (str(project_root / 'assets'), 'assets'),
    
    # Pre-trained AI models
    (str(project_root / 'models'), 'models'),
    
    # Configuration files
    (str(project_root / 'config.json'), '.'),
    (str(project_root / 'settings.json'), '.'),
]

# Hidden imports for modules that PyInstaller might miss
hidden_imports = [
    # Core AI/ML libraries
    'stable_baselines3',
    'stable_baselines3.common.env_util',
    'stable_baselines3.common.callbacks',
    'stable_baselines3.common.vec_env',
    'stable_baselines3.ppo',
    
    # Gymnasium/Gym
    'gymnasium',
    'gymnasium.spaces',
    'gymnasium.envs',
    
    # PyTorch
    'torch',
    'torch.nn',
    'torch.optim',
    'torch.distributions',
    
    # Game engine
    'pygame',
    'pygame.mixer',
    'pygame.font',
    'pygame.image',
    'pygame.transform',
    
    # Scientific computing
    'numpy',
    'scipy',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_agg',
    
    # AI Platform Trainer modules
    'ai_platform_trainer',
    'ai_platform_trainer.ai',
    'ai_platform_trainer.ai.missile_ai_loader',
    'ai_platform_trainer.ai.training',
    'ai_platform_trainer.ai.training.train_missile_rl',
    'ai_platform_trainer.ai.models',
    'ai_platform_trainer.ai.models.missile_model',
    'ai_platform_trainer.entities',
    'ai_platform_trainer.entities.smart_missile',
    'ai_platform_trainer.entities.missile',
    'ai_platform_trainer.entities.enemy_learning',
    'ai_platform_trainer.gameplay',
    'ai_platform_trainer.gameplay.game',
    'ai_platform_trainer.gameplay.game_core',
    'ai_platform_trainer.core',
    'ai_platform_trainer.core.config_manager',
]

# Binaries and collections to include
binaries = []

# Collect all for specific packages
collect_all_packages = [
    'stable_baselines3',
    'gymnasium',
    'torch',
    'pygame',
]

# Build the Analysis
a = Analysis(
    [str(main_script)],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=data_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude development and testing modules
        'pytest',
        'pytest_cov',
        'coverage',
        'mypy',
        'pylint',
        'black',
        'flake8',
        
        # Exclude unnecessary matplotlib backends
        'matplotlib.backends.backend_gtk3agg',
        'matplotlib.backends.backend_gtk3cairo',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_tkagg',
        
        # Exclude tkinter (not used)
        'tkinter',
        'tkinter.ttk',
        
        # Exclude IPython/Jupyter (not used)
        'IPython',
        'jupyter',
        'notebook',
        
        # Exclude large unused scientific libraries
        'scipy.stats',
        'scipy.sparse.csgraph._validation',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Add collect_all for specified packages
for package in collect_all_packages:
    a.datas += collect_all(package)[0]
    a.binaries += collect_all(package)[1]
    a.hiddenimports += collect_all(package)[2]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AI-Platform-Trainer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Use UPX compression to reduce size
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Windowed application (no console)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Application icon (if available)
    icon=str(project_root / 'assets' / 'sprites' / 'player.png') if (project_root / 'assets' / 'sprites' / 'player.png').exists() else None,
)

# For macOS, create an app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='AI-Platform-Trainer.app',
        icon=str(project_root / 'assets' / 'sprites' / 'player.png') if (project_root / 'assets' / 'sprites' / 'player.png').exists() else None,
        bundle_identifier='com.aiplatformtrainer.game',
        info_plist={
            'CFBundleName': 'AI Platform Trainer',
            'CFBundleDisplayName': 'AI Platform Trainer',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'CFBundlePackageType': 'APPL',
            'CFBundleSignature': 'AIPT',
            'NSHighResolutionCapable': 'True',
            'NSSupportsAutomaticGraphicsSwitching': 'True',
        },
    )