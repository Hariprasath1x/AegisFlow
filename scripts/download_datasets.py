#!/usr/bin/env python
"""Download and stage AegisFlow raw datasets into data/raw/."""
from __future__ import annotations
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw'
KAGGLE_SETS = {
    'paysim': 'ealaxi/paysim1',
    'ieee-cis': 'ieee-fraud-detection',
}
DIRECT = {
    'ofac/sdn.csv': 'https://www.treasury.gov/ofac/downloads/sdn.csv',
}
def have_kaggle() -> bool:
    return bool(os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY'))

def fetch_kaggle(name: str, ref: str) -> None:

    target = RAW / name
    if target.exists() and any(target.iterdir()):
        print(f'[skip] {name} already present')
        return
    target.mkdir(parents=True, exist_ok=True)
    cmd = ['kaggle', 'datasets', 'download', '-d', ref, '-p', str(target)]

    if '/' not in ref:
        cmd = ['kaggle', 'competitions', 'download', '-c', ref, '-p', str(target)]
    print(f'[get ] {name} <- {ref}')

    subprocess.run(cmd, check=True)
    for z in target.glob('*.zip'):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(target)
    z.unlink()
