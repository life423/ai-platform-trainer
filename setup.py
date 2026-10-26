import os

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as readme:
    long_description = readme.read() if os.path.exists("README.md") else ""

setup(
    name="ai_platform_trainer",
    version="0.1.0",
    author="AI Platform Trainer Team",
    description="A 2D game about reinforcement learning game AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    entry_points={"console_scripts": ["ai-trainer = ai_platform_trainer.cli:main"]},
    install_requires=[
        "pygame>=2.6.1",
        "torch>=2.7.0",
        "numpy>=2.2.6",
        "gymnasium>=1.1.1",
        "stable-baselines3>=2.6.0",
    ],
    extras_require={
        "dev": [
            "black>=23.1.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pre-commit>=3.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
    ],
    python_requires=">=3.10",
    zip_safe=False,
)
