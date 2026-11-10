"""Colab preparation helpers. Importing this module never starts training.

The notebook embeds this file, while the detector/trainer are checked out at
SOURCE_REV. No credentials, model weights or training data are embedded.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

SOURCE_REV = "d6c1f07dc35e85e5f077b90f7f9b8c7e9a8d62d6"
REPO_URL = "https://github.com/hurry211/MambaCenter.git"
R1_LAST_SHA = "7ac0169203b51ca35419b651d340d4b7ec2ffb583c150f66f822f939c7744b12"
R1_ZIP_SHA = "305df2b5df4121aa88f167819499ec242df2270d11d18c13b16da9b3699ffe55"
PRETRAIN_SHA = "c05e55b2530de061fba26484a73f765979c19217cff5f3b2748b3103415948a6"
TASKS = {
    "R1_RESUME": ("kaggle_addon_vss.yaml", "vss_colab_t4_s317", "dlavss_34"),
    "R2_FRESH": ("kaggle_addon_cp_b1.yaml", "cp_b1_colab_t4_s317", "dlacp_34"),
    "R0_BASELINE": ("kaggle_centernet_dla34.yaml", "centernet_colab_t4_s317", "dla_34"),
    "R3_CPMAMBA": ("kaggle_addon_cp.yaml", "cpmamba_colab_t4_s317", "dlacp_34"),
}
DATA = {
    "train": (6471, 343204, 1549875511,
              "f0a65c7ed3356b37081933e02a970c3b25a13b0b2a6045751106261b888b1c8d"),
    "val": (548, 38759, 81638851,
            "f8aa2e68c27343d0931e0feeda0157ee6b23816199250af0d861a7d15475b4a8"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}\nexpected {expected}\nactual   {actual}")
    return actual


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(pending, path)


def checked(argv, cwd=None, env=None):
    """Forward child stdout/stderr to IPython and preserve full setup diagnostics."""
    args = list(map(str, argv))
    print("RUN:", " ".join(args), flush=True)
    drive_root = globals().get("DRIVE_ROOT")
    log = None
    if drive_root is not None:
        directory = Path(drive_root) / "setup_logs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"command_{time.time_ns()}.log"
        log = path.open("w", encoding="utf-8")
        print("LOG:", path, flush=True)
    process = None
    tail = []
    try:
        process = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            if log is not None:
                log.write(line)
                log.flush()
            tail.append(line)
            tail = tail[-30:]
        returncode = process.wait()
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()
        if log is not None:
            log.close()
    if returncode:
        raise subprocess.CalledProcessError(returncode, args, output="".join(tail))


def native_build_script(workdir):
    """Adapt only the compiler standard; preserve the pinned training source."""
    import torch
    original = Path(workdir) / "runtime/selective-scan/build.py"
    header = Path(torch.__file__).parent / "include/ATen/ATen.h"
    cxx20 = "C++20 or later" in header.read_text(encoding="utf-8")
    standard = "c++20" if cxx20 else "c++17"
    source = original.read_text(encoding="utf-8")
    script = original
    if cxx20:
        if source.count("-std=c++17") != 2:
            raise RuntimeError("Unexpected native build source; review compiler flags before continuing.")
        script = original.with_name("build_colab_cxx20.py")
        script.write_text(source.replace("-std=c++17", "-std=c++20"), encoding="utf-8")
    return script, {"cxx_standard": standard, "original_sha256": sha256(original),
                    "compat_sha256": sha256(script), "max_jobs": 1}


def prepare_repo(workdir):
    """Clone before creating any output directory; reuse only the pinned tree."""
    workdir = Path(workdir)
    if not (workdir / ".git").is_dir():
        if workdir.exists() and any(workdir.iterdir()):
            raise RuntimeError(f"Non-repository directory is not empty: {workdir}. Choose a new WORKDIR.")
        workdir.parent.mkdir(parents=True, exist_ok=True)
        checked(["git", "clone", "--no-checkout", REPO_URL, workdir])
        checked(["git", "-C", workdir, "checkout", "--detach", SOURCE_REV])
    actual = subprocess.check_output(["git", "-C", str(workdir), "rev-parse", "HEAD"], text=True).strip()
    if actual != SOURCE_REV:
        raise RuntimeError(f"Unexpected source revision {actual}; expected {SOURCE_REV}.")
    dirty = subprocess.check_output(["git", "-C", str(workdir), "diff", "--name-only", "HEAD"], text=True)
    if dirty.strip():
        raise RuntimeError(f"Tracked source files changed; not starting an unregistered run:\n{dirty}")
    return actual


def safe_extract(archive, target):
    """Do not allow a downloaded ZIP to write outside its extraction directory."""
    target = Path(target).resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename.replace("\\", "/")
            dest = (target / name).resolve()
            if Path(name).is_absolute() or not dest.is_relative_to(target):
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError(f"Archive symlink is not allowed: {member.filename}")
        z.extractall(target)


def download(url, destination, expected_size=None, expected_sha=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise RuntimeError(f"Incomplete cached file: {destination}. Remove this file explicitly and retry.")
        if expected_sha:
            require_hash(destination, expected_sha)
        return
    temporary = destination.with_name(destination.name + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "MambaCenter-Colab/1"})
    total = 0
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        while block := response.read(8 * 1024 * 1024):
            output.write(block)
            total += len(block)
            if total % (64 * 1024 * 1024) == 0:
                print(f"Downloaded {total / 2**20:.0f} MiB in {time.monotonic()-started:.0f}s", flush=True)
    if expected_size is not None and total != expected_size:
        raise RuntimeError(f"Download size mismatch: {total} != {expected_size}")
    if expected_sha:
        require_hash(temporary, expected_sha)
    os.replace(temporary, destination)


def ensure_pretrained(input_dir):
    import torch
    cache = Path(torch.hub.get_dir()) / "checkpoints" / "dla34-ba72cf86.pth"
    source = Path(input_dir) / cache.name
    if not cache.exists() and source.is_file():
        require_hash(source, PRETRAIN_SHA)
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, cache)
    download("https://huggingface.co/timm/dla34.in1k/resolve/main/pytorch_model.bin", cache,
             expected_size=63105229, expected_sha=PRETRAIN_SHA)
    return {"file": str(cache), "sha256": sha256(cache)}


def coco_fingerprint(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    core = {key: sorted(data[key], key=lambda value: value["id"])
            for key in ("images", "annotations", "categories")}
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return {"n_images": len(core["images"]), "n_annotations": len(core["annotations"]),
            "sha256": hashlib.sha256(canonical).hexdigest()}


def prepare_data(workdir):
    workdir = Path(workdir)
    root = workdir / "data" / "visdrone"
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 6 * 2**30:
        raise RuntimeError("At least 6 GiB of runtime disk space is required for VisDrone.")
    for split, (n_images, n_annotations, archive_size, expected_sha) in DATA.items():
        directory = f"VisDrone2019-DET-{split}"
        image_dir = root / directory / "images"
        annotation_dir = root / directory / "annotations"
        if not image_dir.is_dir() or len(list(image_dir.glob("*.jpg"))) != n_images:
            archive = root / f"{directory}.zip"
            download(f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{directory}.zip",
                     archive, expected_size=archive_size)
            safe_extract(archive, root)
        if len(list(image_dir.glob("*.jpg"))) != n_images or not annotation_dir.is_dir():
            raise RuntimeError(f"Incomplete VisDrone {split} images/annotations.")
    checked([sys.executable, "tools/visdrone2coco.py", "--root", root, "--splits", "train", "val"],
            cwd=workdir)
    fingerprints = {}
    for split, (n_images, n_annotations, _, expected_sha) in DATA.items():
        value = coco_fingerprint(root / f"visdrone2019_det_{split}_coco.json")
        if value != {"n_images": n_images, "n_annotations": n_annotations, "sha256": expected_sha}:
            raise RuntimeError(f"Annotation protocol differs from saved Kaggle R1: {split}: {value}")
        fingerprints[split] = value
    return fingerprints


def assemble_r1_bundle(input_dir, destination):
    """Accept the exact original ZIP or reconstruct its four verified Drive parts."""
    input_dir, destination = Path(input_dir), Path(destination)
    if destination.exists():
        require_hash(destination, R1_ZIP_SHA)
        return destination
    original = input_dir / "r1_epoch8_checkpoint.zip"
    if original.exists():
        require_hash(original, R1_ZIP_SHA)
        return original
    manifest_path = input_dir / "r1_epoch8_checkpoint.manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"R1 input package is missing from {input_dir}. Mount the Google account containing MambaCenter_Colab_Inputs.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != R1_ZIP_SHA or manifest.get("filename") != destination.name:
        raise RuntimeError("Unexpected R1 package manifest.")
    parts = manifest.get("parts", [])
    if not parts or len(parts) != 4:
        raise RuntimeError("The R1 package requires all four parts.")
    pending = destination.with_name(destination.name + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with pending.open("wb") as result:
        for number, part in enumerate(parts, 1):
            expected_name = f"r1_epoch8_checkpoint.zip.part{number:03d}"
            if part.get("name") != expected_name:
                raise RuntimeError("Unexpected checkpoint part name or order.")
            source = input_dir / expected_name
            if source.stat().st_size != part["size"]:
                raise RuntimeError(f"Incomplete checkpoint part: {source}")
            require_hash(source, part["sha256"])
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, result, length=4 * 1024 * 1024)
    require_hash(pending, R1_ZIP_SHA)
    os.replace(pending, destination)
    return destination


def history_records(path):
    records = []
    if Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    epochs = [record["epoch"] for record in records]
    if epochs != sorted(set(epochs)):
        raise RuntimeError("Duplicate or unordered history epochs; preserving files for review.")
    return records


def validate_run(run, task, original=False):
    import torch
    import yaml
    run = Path(run)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    expected_arch = TASKS[task][2]
    expected = {"arch": expected_arch, "seed": 317, "input_res": 960}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise RuntimeError(f"Wrong run configuration: {key}={cfg.get(key)}, expected {value}")
    for key, value in {"batch_size": 8, "accum_steps": 2, "amp": "fp16", "num_workers": 2}.items():
        if cfg["train"].get(key) != value:
            raise RuntimeError(f"Unexpected training recipe: {key}")
    if task == "R2_FRESH" and cfg.get("model", {}).get("cp", {}).get("use_cp_block") is not False:
        raise RuntimeError("R2 must disable the CP block.")
    ckpt_path = run / "model_last.pth"
    if original:
        require_hash(ckpt_path, R1_LAST_SHA)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or not (1 <= epoch <= 15):
        raise RuntimeError(f"Unexpected checkpoint epoch {epoch}")
    if original and (epoch != 8 or checkpoint.get("step") != 3232):
        raise RuntimeError("The original R1 checkpoint must be epoch 8, step 3232.")
    if not checkpoint.get("state_dict") or not checkpoint.get("optimizer") or not checkpoint.get("scaler"):
        raise RuntimeError("Resume requires model, optimizer and FP16 scaler states.")
    records = history_records(run / "history.jsonl")
    if not records or records[-1]["epoch"] != epoch:
        raise RuntimeError(
            f"Checkpoint epoch {epoch} and completed history disagree. The interruption may have occurred "
            "during validation. Preserve this directory and recover the missing evaluation before continuing; "
            "do not silently restart or overwrite it.")
    if records[-1].get("step") != checkpoint.get("step"):
        raise RuntimeError("Checkpoint/history step mismatch.")
    return {"epoch": epoch, "step": checkpoint["step"], "arch": expected_arch,
            "checkpoint_sha256": sha256(ckpt_path)}


def run_identity(run, task):
    run = Path(run)
    return run.parent / "run_identities" / (run.name + ".json"), {
        "task": task, "arch": TASKS[task][2], "source_revision": SOURCE_REV,
        "origin_checkpoint_sha256": R1_LAST_SHA if task == "R1_RESUME" else None,
    }


def prepare_run(run, task, input_dir, bundle_path):
    run = Path(run)
    identity_path, expected_identity = run_identity(run, task)
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != expected_identity:
        raise RuntimeError("Run identity does not match this task/source/origin checkpoint.")
    if (run / "model_last.pth").exists():
        result = validate_run(run, task)
        if not identity_path.exists():
            if task != "R1_RESUME" or result["epoch"] != 8:
                raise RuntimeError("Existing run has no registered source/origin identity; it is not adopted automatically.")
            validate_run(run, task, original=True)
            write_json(identity_path, expected_identity)
        return result
    if run.exists() and any(run.iterdir()):
        raise RuntimeError(f"Existing incomplete run is preserved: {run}. Do not start over in it.")
    if task != "R1_RESUME":
        write_json(identity_path, expected_identity)
        return None  # The trainer creates an empty new run directory itself.
    archive = assemble_r1_bundle(input_dir, bundle_path)
    expected_members = {"model_last.pth", "model_best.pth", "history.jsonl", "config.yaml", "log.txt"}
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != expected_members or len(z.namelist()) != len(expected_members):
            raise RuntimeError("The R1 bundle has unexpected or duplicate members.")
    with tempfile.TemporaryDirectory(prefix="mambacenter-r1-import-") as temporary:
        staging = Path(temporary)
        safe_extract(archive, staging)
        result = validate_run(staging, task, original=True)
        run.mkdir(parents=True, exist_ok=True)
        for source in staging.iterdir():
            shutil.copy2(source, run / source.name)
        validate_run(run, task, original=True)
    write_json(identity_path, expected_identity)
    return result


def launch_training(workdir, run, task, environment, session_file):
    """Runs on the remote Colab runtime, writing every epoch directly to Drive."""
    workdir, run = Path(workdir), Path(run)
    resume = (run / "model_last.pth").exists()
    state = validate_run(run, task) if resume else None
    if state and state["epoch"] == 15:
        print("This run has already completed epoch 15. No training is started.")
        return
    identity_path, expected_identity = run_identity(run, task)
    if not identity_path.exists() or json.loads(identity_path.read_text(encoding="utf-8")) != expected_identity:
        raise RuntimeError("Run prepare_run first; a verified task/source/origin identity is required.")
    arguments = [sys.executable, "tools/train.py"]
    if resume:
        arguments += ["--resume", "--out", str(run)]
    else:
        arguments += ["--config", f"configs/{TASKS[task][0]}", "--out", str(run)]
    arguments += ["--data-dir", str(workdir / "data"), "--set", "train.epochs=15", "eval.interval=1"]
    metadata = {"task": task, "source_revision": SOURCE_REV, "resume": state,
                "command": arguments, "environment": environment,
                "comparison_limit": "Cross-session exploratory result; historical RNG/worker state is unavailable.",
                "storage": "Per-epoch checkpoints and logs written directly to mounted Google Drive."}
    write_json(session_file, metadata)
    print("Outputs:", run)
    print("Interrupted epochs may need to be repeated. Free Colab duration/GPU availability is not guaranteed.")
    checked(arguments, cwd=workdir, env=os.environ.copy())

