#!/usr/bin/env python3
"""Assemble the GLM-5.3 uniform-K3 archive into a vLLM checkpoint.

The published archive contains complete EXL3 routed-expert payloads but keeps
the official BF16 non-routed tensors by reference.  This materializer mirrors
the already-qualified GLM-5.3 EXL3 checkpoint layout: native non-routed
tensors, standard per-expert EXL3 tensor names, embedded/external EXL3 loader
metadata, and a complete safetensors index.

Only the 1,217 non-routed payload ranges are fetched from the pinned official
checkpoint.  Routed BF16 payloads are never downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import time
from typing import Any, BinaryIO, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


SOURCE_REPO = "zai-org/GLM-5.3-BF16"
SOURCE_REVISION = "304b8051cfb2b260b61ce0cbe330e02a98e73639"
MCG_MULTIPLIER = 0xCBAC1FED
ROUTED_WEIGHT = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?:gate_proj|up_proj|down_proj)\.weight$"
)
ARCHIVE_TENSOR = re.compile(
    r"^expert-(?P<expert>\d{3})\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<component>trellis|suh|svh|mcg)$"
)
TORCH_DTYPES = {
    "I16": "torch.int16",
    "I32": "torch.int32",
    "F16": "torch.float16",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(32 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("wb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def padded_header(value: dict[str, Any]) -> bytes:
    encoded = canonical_json(value)
    return encoded + b" " * ((-len(encoded)) % 8)


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        header_bytes = struct.unpack("<Q", raw)[0]
        if header_bytes <= 1 or header_bytes > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(handle.read(header_bytes))
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return header, 8 + header_bytes


def tensor_items(header: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for name, record in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(record, dict):
            raise ValueError(f"invalid tensor record: {name}")
        yield name, record


def archive_manifest_hashes(archive: Path) -> dict[str, str]:
    manifest = read_json(archive / "materialized-uniform-k3" / "MANIFEST.json")
    result = {
        str(row["path"]): str(row["sha256"])
        for row in manifest["files"]
        if str(row["path"]).startswith("layer-")
        and str(row["path"]).endswith(".safetensors")
    }
    mtp_receipt = read_json(archive / "mtp78-k5" / "mtp78-k5.receipts.json")
    result["mtp78-k5.safetensors"] = str(mtp_receipt["consolidated_sha256"])
    return result


def archive_sources(archive: Path) -> list[tuple[int, Path, str]]:
    hashes = archive_manifest_hashes(archive)
    result = []
    for layer in range(3, 78):
        relative = f"layer-{layer:03d}.safetensors"
        source = archive / "materialized-uniform-k3" / relative
        result.append((layer, source, hashes[relative]))
    result.append(
        (
            78,
            archive / "mtp78-k5" / "mtp78-k5.safetensors",
            hashes["mtp78-k5.safetensors"],
        )
    )
    return result


def remap_archive_header(
    source_header: dict[str, Any], layer: int, source_sha256: str
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    output: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "codec": "exl3-mcg",
            "source_archive_sha256": source_sha256,
            "layer": str(layer),
        }
    }
    weight_map: dict[str, str] = {}
    logical: dict[str, dict[str, Any]] = {}
    for source_name, record in tensor_items(source_header):
        match = ARCHIVE_TENSOR.fullmatch(source_name)
        if match is None:
            raise ValueError(f"unexpected archive tensor name: {source_name}")
        expert = int(match.group("expert"))
        projection = match.group("projection")
        component = match.group("component")
        prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        target_name = f"{prefix}.{component}"
        if target_name in output:
            raise ValueError(f"duplicate remapped tensor: {target_name}")
        output[target_name] = record
        logical.setdefault(prefix, {})[component] = record

    storage: dict[str, Any] = {}
    for prefix, components in logical.items():
        if set(components) != {"trellis", "suh", "svh", "mcg"}:
            raise ValueError(f"incomplete EXL3 logical matrix: {prefix}")
        trellis = components["trellis"]
        bits = int(trellis["shape"][2]) // 16
        stored = {}
        for component, record in sorted(components.items()):
            start, stop = (int(value) for value in record["data_offsets"])
            dtype = str(record["dtype"])
            stored[f"{prefix}.{component}"] = {
                "dtype": TORCH_DTYPES[dtype],
                "n_bytes": stop - start,
                "shape": list(record["shape"]),
            }
        storage[prefix] = {
            "bits_per_weight": bits,
            "mcg_multiplier": MCG_MULTIPLIER,
            "quant_format": "exl3",
            "stored_tensors": stored,
        }
    return output, weight_map, storage


def validate_existing_rekeyed(
    destination: Path, expected_names: set[str], expected_payload_bytes: int
) -> None:
    header, data_start = read_safetensors_header(destination)
    names = {name for name, _ in tensor_items(header)}
    if names != expected_names or destination.stat().st_size - data_start != expected_payload_bytes:
        raise ValueError(f"existing rekeyed shard differs: {destination}")


def write_rekeyed_archive_shard(
    source: Path,
    destination: Path,
    layer: int,
    expected_source_sha256: str,
) -> tuple[dict[str, str], dict[str, Any], int]:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"archive shard is absent or unsafe: {source}")
    actual_source_sha256 = sha256_file(source)
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError(f"archive shard SHA-256 differs: {source}")
    source_header, source_data_start = read_safetensors_header(source)
    output_header, _, storage = remap_archive_header(
        source_header, layer, expected_source_sha256
    )
    encoded = padded_header(output_header)
    payload_bytes = source.stat().st_size - source_data_start
    expected_names = {name for name, _ in tensor_items(output_header)}
    weight_map = {name: destination.name for name in expected_names}
    if destination.exists():
        validate_existing_rekeyed(destination, expected_names, payload_bytes)
        return weight_map, storage, payload_bytes

    temporary = destination.with_name(f".{destination.name}.partial")
    with source.open("rb") as inp, temporary.open("wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        inp.seek(source_data_start)
        shutil.copyfileobj(inp, out, length=32 << 20)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, destination)
    validate_existing_rekeyed(destination, expected_names, payload_bytes)
    return weight_map, storage, payload_bytes


def nonrouted_entries(inventory: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries = []
    for name, record in inventory["entries"].items():
        if ROUTED_WEIGHT.fullmatch(name):
            continue
        entries.append((str(name), dict(record)))
    entries.sort(key=lambda item: (str(item[1]["shard"]), int(item[1]["payload_start"])))
    return entries


def group_native_entries(
    entries: list[tuple[str, dict[str, Any]]], max_bytes: int
) -> list[list[tuple[str, dict[str, Any]]]]:
    groups: list[list[tuple[str, dict[str, Any]]]] = []
    current: list[tuple[str, dict[str, Any]]] = []
    current_bytes = 0
    for item in entries:
        size = int(item[1]["nbytes"])
        if current and current_bytes + size > max_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += size
    if current:
        groups.append(current)
    return groups


def source_url(shard: str) -> str:
    return (
        f"https://huggingface.co/{SOURCE_REPO}/resolve/{SOURCE_REVISION}/"
        f"{quote(shard)}"
    )


def stream_range(
    output: BinaryIO,
    *,
    shard: str,
    start: int,
    stop: int,
    expected_sha256: str,
    attempts: int = 4,
) -> None:
    length = stop - start
    if length <= 0:
        raise ValueError("source payload range is empty")
    output_start = output.tell()
    for attempt in range(1, attempts + 1):
        output.seek(output_start)
        output.truncate()
        digest = hashlib.sha256()
        written = 0
        request = Request(
            source_url(shard),
            headers={
                "Range": f"bytes={start}-{stop - 1}",
                "User-Agent": "glm53-k3-checkpoint-materializer/1",
            },
        )
        try:
            with urlopen(request, timeout=120) as response:
                status = getattr(response, "status", None)
                content_range = response.headers.get("Content-Range", "")
                if status != 206 or not content_range.startswith(
                    f"bytes {start}-{stop - 1}/"
                ):
                    raise ValueError(
                        f"range response differs: status={status} range={content_range!r}"
                    )
                while chunk := response.read(16 << 20):
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
            if written != length or digest.hexdigest() != expected_sha256:
                raise ValueError(
                    f"payload closure differs for {shard}:{start}-{stop}: "
                    f"bytes={written}/{length} sha256={digest.hexdigest()}"
                )
            return
        except (HTTPError, URLError, TimeoutError, ValueError, OSError):
            if attempt == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def native_header(
    entries: list[tuple[str, dict[str, Any]]], shard_index: int
) -> tuple[dict[str, Any], int]:
    header: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "origin": "official_bf16_native_range_copy",
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "shard_index": str(shard_index),
        }
    }
    offset = 0
    for name, record in entries:
        size = int(record["nbytes"])
        header[name] = {
            "dtype": str(record["dtype"]),
            "shape": list(record["shape"]),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    return header, offset


def validate_existing_native(
    destination: Path,
    entries: list[tuple[str, dict[str, Any]]],
    payload_bytes: int,
) -> None:
    header, data_start = read_safetensors_header(destination)
    if (
        [name for name, _ in tensor_items(header)] != [name for name, _ in entries]
        or destination.stat().st_size - data_start != payload_bytes
    ):
        raise ValueError(f"existing native shard differs: {destination}")


def write_native_shard(
    destination: Path,
    entries: list[tuple[str, dict[str, Any]]],
    shard_index: int,
) -> tuple[dict[str, str], int]:
    header, payload_bytes = native_header(entries, shard_index)
    encoded = padded_header(header)
    if destination.exists():
        validate_existing_native(destination, entries, payload_bytes)
        return {name: destination.name for name, _ in entries}, payload_bytes

    temporary = destination.with_name(f".{destination.name}.partial")
    with temporary.open("wb+") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for tensor_index, (name, record) in enumerate(entries, 1):
            print(
                f"native shard {shard_index}: tensor {tensor_index}/{len(entries)} "
                f"{name} ({int(record['nbytes'])} bytes)",
                flush=True,
            )
            stream_range(
                out,
                shard=str(record["shard"]),
                start=int(record["payload_start"]),
                stop=int(record["payload_end"]),
                expected_sha256=str(record["payload_sha256"]),
            )
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, destination)
    validate_existing_native(destination, entries, payload_bytes)
    return {name: destination.name for name, _ in entries}, payload_bytes


def copy_auxiliary(aux_root: Path, output: Path) -> None:
    for name in (
        ".gitattributes",
        "LICENSE",
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        source = aux_root / name
        destination = output / name
        if not source.is_file():
            raise ValueError(f"source auxiliary file is missing: {source}")
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise ValueError(f"existing auxiliary file differs: {destination}")
            continue
        shutil.copy2(source, destination)


def build_quantization_config(tensor_storage: dict[str, Any]) -> dict[str, Any]:
    return {
        "bits": 3,
        "codebook": "mcg",
        "head_bits": 16,
        "non_routed_dtype_policy": "official_source_native",
        "quant_method": "exl3",
        "scope": "glm53_routed_experts_layers_3_78",
        "serving_reader_qualified": False,
        "tensor_storage": dict(sorted(tensor_storage.items())),
        "version": "0.0.43",
        "r7_routed_experts": {
            "schema": "r7-complete-v2-checkpoint-v1",
            "codebook": "mcg",
            "bits": "mixed_tensor",
            "moe_layers": [3, 78],
            "k_values": [3, 5],
            "target_bpw": "3.0",
            "mtp_layer_78": "uniform_k5",
            "tp_slice_quantum": 128,
        },
    }


def materialize(args: argparse.Namespace) -> None:
    archive = args.archive.resolve()
    aux_root = args.aux_root.resolve()
    inventory_path = archive / "evidence" / "source-inventory.json"
    inventory = read_json(inventory_path)
    if (
        inventory.get("config_sha256")
        != sha256_file(aux_root / "config.json")
        or inventory.get("index_sha256")
        != sha256_file(aux_root / "model.safetensors.index.json")
    ):
        raise ValueError("pinned BF16 auxiliary identity differs from source inventory")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    copy_auxiliary(aux_root, output)

    native = nonrouted_entries(inventory)
    native_bytes = sum(int(record["nbytes"]) for _, record in native)
    if len(native) != 1217 or native_bytes != 37_781_104_640:
        raise ValueError(
            f"non-routed inventory differs: tensors={len(native)} bytes={native_bytes}"
        )
    groups = group_native_entries(native, args.native_shard_bytes)
    if args.native_only_shard is not None:
        index = args.native_only_shard
        if index < 1 or index > len(groups):
            raise ValueError(
                f"native shard index must be from 1 to {len(groups)}, got {index}"
            )
        destination = output / f"native-{index:05d}-of-{len(groups):05d}.safetensors"
        _, payload_bytes = write_native_shard(destination, groups[index - 1], index)
        print(
            json.dumps(
                {
                    "native_shard": index,
                    "native_shard_count": len(groups),
                    "payload_bytes": payload_bytes,
                    "complete": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    output_map: dict[str, str] = {}
    tensor_storage: dict[str, Any] = {}
    total_payload_bytes = 0
    sources = archive_sources(archive)
    for layer, source, expected_sha256 in sources:
        destination = output / f"exl3-r7-layer-{layer:03d}.safetensors"
        print(f"rekeying layer {layer}: {source.name}", flush=True)
        layer_map, layer_storage, payload_bytes = write_rekeyed_archive_shard(
            source, destination, layer, expected_sha256
        )
        if set(output_map).intersection(layer_map):
            raise ValueError(f"duplicate EXL3 tensor names in layer {layer}")
        output_map.update(layer_map)
        tensor_storage.update(layer_storage)
        total_payload_bytes += payload_bytes

    for index, group in enumerate(groups, 1):
        destination = output / f"native-{index:05d}-of-{len(groups):05d}.safetensors"
        native_map, payload_bytes = write_native_shard(destination, group, index)
        if set(output_map).intersection(native_map):
            raise ValueError(f"native/EXL3 tensor collision in shard {index}")
        output_map.update(native_map)
        total_payload_bytes += payload_bytes

    quantization = build_quantization_config(tensor_storage)
    config = read_json(aux_root / "config.json")
    config["quantization_config"] = quantization
    atomic_json(output / "quantization_config.json", quantization)
    atomic_json(output / "config.json", config)
    index = {
        "metadata": {"total_size": total_payload_bytes},
        "weight_map": dict(sorted(output_map.items())),
    }
    atomic_json(output / "model.safetensors.index.json", index)
    receipt = {
        "schema": "local-inference-lab.glm53-uniform-k3-runtime-checkpoint.v1",
        "complete": True,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_inventory_sha256": str(inventory["inventory_sha256"]),
        "archive": str(archive),
        "archive_revision": args.archive_revision,
        "tensor_count": len(output_map),
        "exl3_logical_matrix_count": len(tensor_storage),
        "nonrouted_tensor_count": len(native),
        "nonrouted_payload_bytes": native_bytes,
        "total_payload_bytes": total_payload_bytes,
        "config_sha256": sha256_file(output / "config.json"),
        "quantization_config_sha256": sha256_file(
            output / "quantization_config.json"
        ),
        "index_sha256": sha256_file(output / "model.safetensors.index.json"),
        "routed_layers": [3, 78],
        "main_bits": 3,
        "mtp78_bits": 5,
        "runtime_tp": 4,
        "serving_reader_qualified": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    atomic_json(output / "materialization-receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--aux-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--archive-revision",
        default="4576d6014f417ddc241d58e4aadf9cba6c5b4f07",
    )
    parser.add_argument(
        "--native-shard-bytes", type=int, default=4_000_000_000
    )
    parser.add_argument("--native-only-shard", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    materialize(parse_args())
