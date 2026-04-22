"""
Module A main entry point.

Competition mode:
  - default source is the local `test_files` folder
  - URL mode is preserved in comments for quick restoration later
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_env():
    for env_file in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            return


def load_local_files() -> list[Path]:
    from config import (
        SUPPORTED_AUDIO_TYPES,
        SUPPORTED_IMAGE_TYPES,
        SUPPORTED_TEXT_TYPES,
        SUPPORTED_VIDEO_TYPES,
        TEST_FILES_DIR,
    )

    supported_ext = SUPPORTED_TEXT_TYPES | SUPPORTED_IMAGE_TYPES | SUPPORTED_VIDEO_TYPES | SUPPORTED_AUDIO_TYPES
    return [
        path for path in TEST_FILES_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_ext
    ]


def run_pipeline(local_files: list[Path], interactive: bool = True):
    from agents.collection import CollectionAgent
    from agents.generation import GenerationAgent
    from agents.moderation import ModerationAgent
    from analysis.features import FeatureExtractor
    from analysis.statistics import run_analysis
    from config import ANALYSIS_OUTPUT_DIR
    from database.manager import init_database

    sep = "=" * 60
    print(sep)
    print("  EDUCATIONAL MATERIALS AGENT")
    print(sep)

    print("\n[1/7] Initialising database...")
    init_database()

    if local_files:
        print(f"\n[2/7] Collecting {len(local_files)} local file(s) from test_files...")
        ids = CollectionAgent().process_local_files(local_files)
        print(f"  Collected {len(ids)} material(s).")
    else:
        print("\n[2/7] No local files found in test_files — skipping collection.")

    print("\n[3/7] Running moderation...")
    results = ModerationAgent().moderate_all()
    print(f"  Moderated {len(results)} material(s).")

    fe = FeatureExtractor()
    print("\n[4/7] Extracting features...")
    fe.extract_all_features()

    print("\n[5/7] Computing topic adjacency...")
    fe.compute_topic_adjacency()

    print("\n[6/7] Analysing feature importance...")
    importance = fe.analyse_importance()
    if importance:
        print("  Top-5 features:")
        for item in importance[:5]:
            print(f"    {item['feature_name']}: {item['importance_score']:.3f}")

    print("\n[7/7] Generating reports & visualisations...")
    run_analysis()
    print(f"  Reports in: {ANALYSIS_OUTPUT_DIR.resolve()}")

    if interactive:
        GenerationAgent().run_interactive()

    print(f"\n{sep}")
    print("  Pipeline complete!")
    print(sep)


def main():
    load_env()
    parser = argparse.ArgumentParser(description="Educational Materials Agent")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive generation")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--moderate-only", action="store_true")
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    local_files = load_local_files()

    # URL mode preserved for quick restoration later:
    # parser.add_argument("urls", nargs="*", help="URLs to process")
    # parser.add_argument("-f", "--urls-file", help="File with URLs (one per line)")
    # urls = load_urls(args.urls_file, args.urls)

    if args.collect_only:
        from agents.collection import CollectionAgent
        from database.manager import init_database

        init_database()
        CollectionAgent().process_local_files(local_files)
        return

    if args.moderate_only:
        from agents.moderation import ModerationAgent
        from database.manager import init_database

        init_database()
        ModerationAgent().moderate_all()
        return

    if args.analyse_only:
        from analysis.features import FeatureExtractor
        from analysis.statistics import run_analysis
        from database.manager import init_database

        init_database()
        fe = FeatureExtractor()
        fe.extract_all_features()
        fe.compute_topic_adjacency()
        fe.analyse_importance()
        run_analysis()
        return

    if args.generate:
        from agents.generation import GenerationAgent
        from database.manager import init_database

        init_database()
        GenerationAgent().run_interactive()
        return

    run_pipeline(local_files, interactive=not args.no_interactive)


if __name__ == "__main__":
    main()
