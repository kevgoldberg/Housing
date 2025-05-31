import json
from pathlib import Path
import argparse


def convert_notebook(nb_path: Path, py_path: Path) -> None:
    """Convert a Jupyter notebook to a plain Python script."""
    with nb_path.open() as f:
        nb = json.load(f)

    lines = []
    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type")
        source = cell.get("source", [])
        if cell_type == "markdown":
            lines.append("# " + "-" * 80 + "\n")
            for line in source:
                lines.append("# " + line.rstrip() + "\n")
            lines.append("\n")
        elif cell_type == "code":
            for line in source:
                lines.append(line)
            lines.append("\n\n")

    py_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .ipynb files to .py")
    parser.add_argument("notebooks", nargs="+", type=Path, help="Notebook files")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    for nb in args.notebooks:
        out_dir = args.out_dir if args.out_dir else nb.parent
        py_file = out_dir / (nb.stem + ".py")
        convert_notebook(nb, py_file)
        print(f"Converted {nb} -> {py_file}")


if __name__ == "__main__":
    main()
