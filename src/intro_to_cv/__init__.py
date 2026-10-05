"""Command entry point for the two Part 1 baselines and shared comparison."""


def main() -> None:
    import argparse
    import importlib
    import sys

    commands = {
        "template": "intro_to_cv.baselines.template.cli",
        "template-eval": "intro_to_cv.baselines.template.evaluate",
        "template-maps": "intro_to_cv.baselines.template.maps",
        "autoencoder": "intro_to_cv.baselines.autoencoder.cli",
        "autoencoder-eval": "intro_to_cv.baselines.autoencoder.evaluate",
        "compare": "intro_to_cv.experiments.compare",
    }
    parser = argparse.ArgumentParser(description="VisA Part 1 baselines (CPU)")
    parser.add_argument("command", choices=commands)
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    command = sys.argv[1]
    if command not in commands:
        parser.error(f"Unknown command: {command}")
    del sys.argv[1]
    importlib.import_module(commands[command]).main()
