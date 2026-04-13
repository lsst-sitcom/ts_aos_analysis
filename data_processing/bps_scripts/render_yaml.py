#!/usr/bin/env python

import argparse
from jinja2 import Template
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Render a Jinja2-templated YAML file with day_obs substitution."
    )
    parser.add_argument(
        "day_obs",
        type=int,
        help="The day_obs value to substitute (e.g., 20260409)",
    )
    parser.add_argument(
        "-t", "--template",
        type=Path,
        default=Path("template.yaml"),
        help="Path to the Jinja2 template YAML file (default: template.yaml)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to the output YAML file (default: prints to stdout)",
    )
    args = parser.parse_args()

    if not args.template.exists():
        raise FileNotFoundError(f"Template file not found: {args.template}")

    template = Template(args.template.read_text())
    rendered = template.render(day_obs=args.day_obs)

    if args.output:
        args.output.write_text(rendered)
        print(f"Rendered YAML written to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

