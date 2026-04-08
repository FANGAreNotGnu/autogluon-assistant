"""
CLI command to set up a MLZero Claude Code workspace.

Usage:
    mlzero-cc <dest_path> <output_path>

Copies the Claude Code skills template to <dest_path> and configures
<output_path> as the root folder for storing run outputs.
"""

import argparse
import os
import shutil
import sys


def get_template_dir():
    """Get the path to the cc_template directory shipped with the package."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "cc_template")


def setup_workspace(dest_path: str, output_path: str):
    """
    Copy the Claude Code skills template to dest_path and configure output_path.

    Args:
        dest_path: Destination directory for the Claude Code workspace
        output_path: Root directory for storing MLZero run outputs
    """
    template_dir = get_template_dir()

    if not os.path.isdir(template_dir):
        print(f"Error: Template directory not found at {template_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve to absolute paths
    dest_path = os.path.abspath(dest_path)
    output_path = os.path.abspath(output_path)

    # Create dest_path if it doesn't exist
    os.makedirs(dest_path, exist_ok=True)

    # Copy template contents to dest_path
    for item in os.listdir(template_dir):
        src = os.path.join(template_dir, item)
        dst = os.path.join(dest_path, item)

        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # Replace {{OUTPUT_ROOT}} placeholder in all files
    for root, dirs, files in os.walk(dest_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r") as f:
                    content = f.read()
                if "{{OUTPUT_ROOT}}" in content:
                    content = content.replace("{{OUTPUT_ROOT}}", output_path)
                    with open(fpath, "w") as f:
                        f.write(content)
            except (UnicodeDecodeError, PermissionError):
                # Skip binary files or permission issues
                continue

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    print(f"MLZero Claude Code workspace set up at: {dest_path}")
    print(f"Run outputs will be stored under: {output_path}")
    print()
    print("To start using MLZero with Claude Code:")
    print(f"  cd {dest_path}")
    print("  claude")
    print()
    print("Then use the skills:")
    print("  /run-mlzero dataset=/path/to/data instruction=\"Predict X. Metric: Y.\"")


def main():
    parser = argparse.ArgumentParser(
        description="Set up a MLZero Claude Code workspace",
        usage="mlzero-cc <dest_path> <output_path>",
    )
    parser.add_argument("dest_path", help="Destination directory for the Claude Code workspace")
    parser.add_argument("output_path", help="Root directory for storing MLZero run outputs")

    args = parser.parse_args()
    setup_workspace(args.dest_path, args.output_path)


if __name__ == "__main__":
    main()
