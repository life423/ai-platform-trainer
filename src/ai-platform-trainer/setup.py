from setuptools import setup, find_packages

setup(
    name="ai_platform_trainer",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            # "run-game" is the command name you’ll type in Terminal;
            # "ai_platform_trainer.main:main" is the Python function to run.
            "run-game = ai_platform_trainer.main:main"
        ]
    },
)
