# AI Platform Trainer - Deployment Guide

This document describes the automated deployment system that builds and distributes executable files via GitHub Actions.

## 🚀 Automated Deployment

### How It Works

When code is merged to the `main` branch, GitHub Actions automatically:

1. **Pre-trains AI Models**: Trains intelligent missile AI models (75k timesteps)
2. **Builds Executables**: Creates standalone executables for Windows, macOS, and Linux
3. **Creates Release**: Publishes a new GitHub release with all executables
4. **Bundles Smart AI**: Includes pre-trained models so users get intelligent missiles instantly

### Release Artifacts

Each automated release includes:

- **Windows**: `AI-Platform-Trainer-Windows.exe` (standalone executable)
- **macOS**: `AI-Platform-Trainer-macOS.dmg` (disk image) and `AI-Platform-Trainer-macOS` (executable)
- **Linux**: `AI-Platform-Trainer-Linux` (standalone executable)
- **Checksums**: `checksums.txt` (SHA256 hashes for verification)

## 🎯 User Experience

### For End Users

1. **Download**: Get the appropriate file for your OS from GitHub releases
2. **Run**: Double-click the executable - no installation required!
3. **Play**: Missiles are immediately intelligent (no waiting for training)

### Smart AI Out-of-the-Box

- ✅ **Pre-trained Models**: Advanced RL missile AI included
- ✅ **Instant Gameplay**: No first-time setup or training delays
- ✅ **Intelligent Behavior**: Missiles intercept moving targets efficiently
- ✅ **Zero Dependencies**: Everything bundled in the executable

## 🛠️ Development

### Manual Build (for testing)

```bash
# Install dependencies
pip install -r requirements.txt
pip install "stable-baselines3[extra]"

# Pre-train AI models
python scripts/pre_train_for_deployment.py

# Build executable using spec file
pyinstaller ai-platform-trainer.spec
```

### Local Testing

```bash
# Run from source (will trigger first-time training if no models)
python run_game.py

# Test built executable
./dist/AI-Platform-Trainer  # Linux/macOS
./dist/AI-Platform-Trainer.exe  # Windows
```

## 📊 Build Process Details

### Pre-training Phase

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Training Steps**: 75,000 timesteps (~5-10 minutes)
- **Performance Target**: >60% hit rate, >0.6 average reward
- **Model Size**: ~150KB (highly optimized)

### PyInstaller Configuration

- **Mode**: Single-file executable (`--onefile`)
- **GUI**: Windowed application (`--windowed`)
- **Compression**: UPX enabled for smaller size
- **Data Files**: Assets, models, config files bundled
- **Hidden Imports**: All AI/ML dependencies detected

### Cross-Platform Builds

| Platform | Runner | Output | Special Notes |
|----------|--------|---------|---------------|
| Windows | `windows-latest` | `.exe` | Code signing ready |
| macOS | `macos-latest` | `.app` + `.dmg` | App bundle created |
| Linux | `ubuntu-latest` | Binary | AppImage possible |

## 🔧 Configuration Files

### GitHub Actions Workflow
- **File**: `.github/workflows/deploy.yml`
- **Triggers**: Push to main, manual dispatch
- **Artifacts**: Executables + checksums

### PyInstaller Spec
- **File**: `ai-platform-trainer.spec`
- **Configuration**: Build settings, data files, imports
- **Optimization**: Size reduction, dependency management

### Pre-training Script
- **File**: `scripts/pre_train_for_deployment.py`
- **Purpose**: Creates optimized AI models for bundling
- **Output**: `models/missile_rl_model_final.zip`

## 📈 Performance Metrics

### Executable Sizes (Approximate)
- **Windows**: ~200-300 MB
- **macOS**: ~250-350 MB  
- **Linux**: ~200-300 MB

### Build Times (GitHub Actions)
- **Pre-training**: 5-10 minutes
- **Windows Build**: 3-5 minutes
- **macOS Build**: 3-5 minutes
- **Linux Build**: 3-5 minutes
- **Total Pipeline**: ~15-25 minutes

### AI Performance (Bundled Models)
- **Hit Rate**: 70-85% (vs 10-20% basic homing)
- **Interception**: Smart trajectory planning
- **Responsiveness**: Real-time decision making
- **No Training Delay**: Instant intelligent behavior

## 🚦 Release Management

### Versioning
- **Format**: `vYYYY.MM.DD-BUILD_NUMBER`
- **Example**: `v2025.07.25-42`
- **Automatic**: Based on build date and run number

### Release Notes
- **Auto-generated**: Include build info, commit hash, features
- **Performance**: Model metrics and build statistics
- **Instructions**: Download and usage guidance

## 🔍 Troubleshooting

### Build Issues

**Large executable size**: 
- UPX compression enabled
- Unused dependencies excluded
- Consider AppImage for Linux

**Missing dependencies**:
- Check `hidden_imports` in spec file
- Verify `collect_all` packages
- Test import paths

**Model loading failures**:
- Verify bundle paths (`sys._MEIPASS`)
- Check model file inclusion
- Test fallback mechanisms

### User Issues

**Slow startup**:
- Models are pre-loaded (expected ~2-3 seconds)
- GPU detection may add delay
- Consider splash screen

**No intelligent missiles**:
- Check bundled model integrity
- Verify stable-baselines3 inclusion
- Fall back to basic homing

## 🎮 Future Enhancements

- **Auto-updater**: Check for new releases
- **Smaller builds**: More aggressive optimization
- **Code signing**: Windows/macOS app signing
- **App stores**: Distribution via Steam, Microsoft Store
- **Portable mode**: Fully self-contained with settings