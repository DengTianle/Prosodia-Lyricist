"""Install the pinned baseline pronunciation data: python -m prosodia_lyricist.historical_ipa."""

import hashlib
import io
import json
import tarfile
from pathlib import Path
from urllib.request import urlopen

SOURCE_URL = (
    "https://files.pythonhosted.org/packages/50/6e/"
    "b2ed4c85f7d613d0e317142a7f21ec8f9ee3402f8a258c9338d3c0abf50f/prosodic-1.5.0.tar.gz"
)
ARCHIVE_SHA256 = "ed0ae28da809bbc769e0bb95123ece04a5489bebaf0ebee98ad9e0ca205d73bc"
FILES_SHA256 = {
    "english.tsv": "45d64b42f96b8c10f56a47b3dc2cbf6bd7687c25a4b58bc5ac93cf7461dfb85f",
    "maybestressed.txt": "43346ab57238625ea5cb9a92edb710445d13edeae9c5062c3ff5b652434cdce6",
    "unstressed.txt": "4a6558fad74b08c03d9fd3bc94367801fe04dc551c71aacc1020f3c39e5af7a8",
}


def verify_resources(directory):
    try:
        provenance = json.loads(directory.joinpath("provenance.json").read_text(encoding="utf-8"))
        if provenance["version"] != "1.5.0" or provenance["files_sha256"] != FILES_SHA256:
            raise ValueError("Unexpected historical dictionary provenance")
        for name, digest in FILES_SHA256.items():
            if hashlib.sha256(directory.joinpath(name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Historical dictionary checksum differs: {name}")
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(
            "Historical IPA data is missing or differs from Prosodic 1.5.0. Run "
            "python -m prosodia_lyricist.historical_ipa once, then prepare again."
        ) from exc
    return provenance


def install(directory, *, archive_path=None):
    directory = Path(directory)
    # Only exact member bytes are read; no archive paths are extracted to disk.
    if archive_path is None:
        with urlopen(SOURCE_URL, timeout=60) as response:
            data = response.read()
    else:
        data = Path(archive_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Prosodic source archive checksum differs")
    payloads = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for name, digest in FILES_SHA256.items():
            member = archive.getmember(f"prosodic-1.5.0/prosodic/dicts/en/{name}")
            with archive.extractfile(member) as handle:
                payloads[name] = handle.read()
            if hashlib.sha256(payloads[name]).hexdigest() != digest:
                raise ValueError(f"Prosodic dictionary checksum differs: {name}")
        with archive.extractfile("prosodic-1.5.0/LICENSE.md") as handle:
            payloads["LICENSE.upstream.txt"] = handle.read()
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        directory.joinpath(name).write_bytes(value)
    provenance = dict(
        package="prosodic",
        version="1.5.0",
        source_url=SOURCE_URL,
        archive_sha256=ARCHIVE_SHA256,
        files_sha256=FILES_SHA256,
        archive_license_file="LICENSE.upstream.txt",
        archive_license_title="GNU GPL version 3",
        pypi_declared_license="MPL-2.0",
        policy="Dictionary order; maybestressed before unstressed; first variant; modern fallback.",
    )
    directory.joinpath("provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    directory.joinpath("NOTICE.txt").write_text(
        "Unmodified pronunciation data from Prosodic 1.5.0 (Ryan Heuser). Source and SHA256 "
        "checksums are in provenance.json. Original LICENSE.md is retained verbatim. "
        "The XAI-Lyricist authors did not record their exact Prosodic version.\n"
    )
    verify_resources(directory)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Previously downloaded pinned source archive")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "resources/prosodic_1_5",
    )
    args = parser.parse_args()
    target = args.output_dir
    install(target, archive_path=args.archive)
    print(f"Installed verified historical pronunciation data: {target}")
