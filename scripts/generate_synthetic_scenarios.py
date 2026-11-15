#!/usr/bin/env python3
"""CLI utility to generate and verify AegisFlow M2 deterministic synthetic scenarios."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from typing import Sequence

from aegis_ml.scenarios.generator import (
    ManifestGenerator,
    canonical_json_bytes,
    verify_manifest_hash,
)
from aegis_ml.scenarios.models import ScenarioManifest


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic known-fraud scenarios for AegisFlow M2."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Target filepath to export the serialized scenario manifest JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Permit replacing an existing destination file when exporting manifest.",
    )
    parser.add_argument(
        "--verify-hash",
        action="store_true",
        help="Verify the SHA-256 hash of the generated or provided manifest.",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="Existing manifest file to verify against its embedded hash.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override seed (defaults to canonical seed 20261010).",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    opts = parse_args(args)

    try:
        manifest: ScenarioManifest
        if opts.input_json:
            if not os.path.exists(opts.input_json):
                sys.stderr.write(f"Error: input file does not exist: {opts.input_json}\n")
                return 1
            with open(opts.input_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            manifest = ScenarioManifest.model_validate(data)
            sys.stdout.write(f"Loaded manifest from {opts.input_json}\n")
        else:
            gen = ManifestGenerator(seed=opts.seed) if opts.seed is not None else ManifestGenerator()
            manifest = gen.generate()
            sys.stdout.write(
                f"Generated manifest: {len(manifest.scenarios)} scenarios, hash={manifest.manifest_hash}\n"
            )

        if opts.verify_hash:
            if not verify_manifest_hash(manifest):
                sys.stderr.write(
                    f"Error: Manifest hash mismatch! Embedded: {manifest.manifest_hash}\n"
                )
                return 2
            sys.stdout.write("Manifest hash verification: PASSED\n")

        if opts.output_json:
            out_dir = os.path.dirname(opts.output_json)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            payload_str = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)

            if not opts.overwrite:
                # Exclusive creation: fails if destination already exists (protects against TOCTOU)
                try:
                    with open(opts.output_json, "x", encoding="utf-8") as f:
                        f.write(payload_str)
                except FileExistsError:
                    sys.stderr.write(
                        f"Error: destination file already exists: {opts.output_json} "
                        f"(use --overwrite to replace)\n"
                    )
                    return 1
                except Exception:
                    # Clean up partial output if write failed midway
                    if os.path.exists(opts.output_json):
                        try:
                            os.unlink(opts.output_json)
                        except OSError:
                            pass
                    raise
            else:
                # Atomic replacement via temporary file in same directory
                temp_fd, temp_path = tempfile.mkstemp(dir=out_dir or ".", prefix=".manifest_tmp_")
                replaced = False
                try:
                    # Apply appropriate file permissions (preserving existing or standard umask)
                    try:
                        current_umask = os.umask(0)
                        os.umask(current_umask)
                        default_mode = 0o666 & ~current_umask
                        target_mode = (
                            stat.S_IMODE(os.stat(opts.output_json).st_mode)
                            if os.path.exists(opts.output_json)
                            else default_mode
                        )
                        os.fchmod(temp_fd, target_mode)
                    except OSError:
                        pass

                    with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                        f.write(payload_str)
                    os.replace(temp_path, opts.output_json)
                    replaced = True
                finally:
                    if not replaced:
                        try:
                            os.close(temp_fd)
                        except OSError:
                            pass
                        if os.path.exists(temp_path):
                            try:
                                os.unlink(temp_path)
                            except OSError:
                                pass

            sys.stdout.write(f"Manifest exported successfully to: {opts.output_json}\n")

        return 0

    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

