import numpy as np
import argparse
import shutil
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten subset")
    parser.add_argument("-i", "--input", type=str, help="Input file path")
    parser.add_argument("-o", "--output", type=str, help="Output file path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    out_path.mkdir(parents=True, exist_ok=True)

    for idx, file in enumerate(in_path.rglob("*.nii.gz")):
        out_file_path = out_path / file.name.replace("_0000", "")
        shutil.copy(file, out_file_path)
        print(f"{file} ---> {out_file_path}")

    print(f"Done, transfered {idx + 1} files")
