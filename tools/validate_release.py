"""Validate the LGC-Net release package and its BookLayout-Bi interface."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import LGCNet


EXPECTED_SPLITS = {
    "stage1_train": 8_971,
    "stage1_validation": 996,
    "stage2_train": 1_572,
    "stage2_validation": 174,
    "test": 194,
}
FORBIDDEN_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff",
    ".pt", ".pth", ".ckpt", ".onnx", ".pkl", ".npz", ".doc", ".docx",
    ".ppt", ".pptx", ".pdf", ".key", ".pem", ".p12", ".pfx",
}
TEXT_SUFFIXES = {".py", ".md", ".json", ".txt", ".patch", ".gitattributes", ".gitignore"}
SENSITIVE = re.compile(
    r"(?:[A-Za-z]:\\(?:Users|PycharmProjects)\\|/home/|/Users/|"
    r"ghp_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9]{20,}|"
    r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY)"
)


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--booklayout-root",
        type=Path,
        default=Path("../BookLayout-Bi"),
        help="Path to the sibling BookLayout-Bi repository.",
    )
    args = parser.parse_args()

    root = ROOT
    protocol = json.loads((root / "configs" / "final_protocol.json").read_text(encoding="utf-8"))
    assert protocol["protocol_name"] == "M-32-G"
    assert protocol["training_seeds"] == [0, 1, 2, 3, 4]
    assert protocol["expected_split_counts"] == EXPECTED_SPLITS
    assert protocol["model"]["latent_dim"] == 32
    assert protocol["model"]["max_seq_len"] == 15

    reported = json.loads((root / "results" / "reported_main_comparison.json").read_text(encoding="utf-8"))
    methods = {entry["method"]: entry for entry in reported["methods"]}
    assert methods["LGC-Net (M-32-G)"]["iou"] == [0.2027, 0.0044]
    assert methods["LGC-Net (M-32-G)"]["max_iou"] == [0.2500, 0.0028]
    assert methods["LayoutTransformer"]["iou"] == [0.1566, 0.0031]

    model = LGCNet(**protocol["model"])
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    encoder_prefixes = ("box_embed", "posterior_encoder", "to_mu", "to_logvar")
    encoder_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(encoder_prefixes)
    )
    assert total_parameters == 1_436_612
    assert total_parameters - encoder_parameters == 816_132

    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.resolve() == Path(__file__).resolve()
            or ".git" in path.parts
            or "__pycache__" in path.parts
        ):
            continue
        assert path.suffix.lower() not in FORBIDDEN_SUFFIXES, path
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"README.md", ".gitignore", ".gitattributes"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert not SENSITIVE.search(text), path

    booklayout_root = args.booklayout_root.resolve()
    annotation_path = booklayout_root / "data" / "booklayout_bi_annotations.json"
    split_root = booklayout_root / "data" / "splits"
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"BookLayout-Bi annotations not found at {annotation_path}. "
            "Pass --booklayout-root with the correct sibling repository path."
        )
    records = json.loads(annotation_path.read_text(encoding="utf-8"))["records"]
    assert len(records) == 10_161
    by_id = {record["sample_id"]: record for record in records}
    split_ids = {name: read_ids(split_root / f"{name}.txt") for name in EXPECTED_SPLITS}
    assert {name: len(ids) for name, ids in split_ids.items()} == EXPECTED_SPLITS
    long_counts = {
        name: sum(len(by_id[sample_id]["elements"]) > 15 for sample_id in ids)
        for name, ids in split_ids.items()
    }
    assert long_counts == {
        "stage1_train": 21,
        "stage1_validation": 3,
        "stage2_train": 0,
        "stage2_validation": 0,
        "test": 0,
    }
    print(
        "LGC-Net validation: PASS "
        f"(parameters={total_parameters}, deployment_parameters={total_parameters - encoder_parameters})"
    )


if __name__ == "__main__":
    main()
