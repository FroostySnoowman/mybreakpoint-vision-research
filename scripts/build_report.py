"""Build publication PDFs from the repository's Typst sources."""

from pathlib import Path

import typst


def main():
    root = Path(__file__).resolve().parents[1]
    for name in ('report', 'summary'):
        source = root / 'report' / f'{name}.typ'
        output = source.with_suffix('.pdf')
        typst.compile(str(source), output=str(output), root=str(root))
        print(output.relative_to(root))


if __name__ == '__main__':
    main()
