"""Allow `python -m llmtuner` to run the training entry point."""

from .trainer.cli import main

if __name__ == "__main__":
    main()
