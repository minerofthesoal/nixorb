"""nixorb/utils/crypto.py — PBKDF2 + Fernet encrypted config export/import."""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import tarfile
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

log = logging.getLogger(__name__)

_ITERATIONS = 480_000
_SALT_LEN   = 16


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def export_config(settings, output_path: str, password: str = "nixorb") -> None:
    """Serialize settings + memory into an encrypted .tar.gz.enc file."""
    salt      = os.urandom(_SALT_LEN)
    key       = _derive_key(password, salt)
    fernet    = Fernet(key)
    json_data = settings.model_dump_json(indent=2).encode()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info      = tarfile.TarInfo(name="config.json")
        info.size = len(json_data)
        tar.addfile(info, io.BytesIO(json_data))

        mem_dir = Path(settings.memory_dir)
        if mem_dir.exists():
            tar.add(mem_dir, arcname="memory")

    encrypted = salt + fernet.encrypt(buf.getvalue())
    Path(output_path).write_bytes(encrypted)
    log.info("Config exported → %s", output_path)


def import_config(settings, input_path: str, password: str = "nixorb") -> None:
    """Decrypt and restore settings from a .tar.gz.enc file."""
    raw  = Path(input_path).read_bytes()
    salt = raw[:_SALT_LEN]
    blob = raw[_SALT_LEN:]
    key  = _derive_key(password, salt)

    try:
        plaintext = Fernet(key).decrypt(blob)
    except InvalidToken:
        raise ValueError("Wrong password or corrupted archive") from None

    buf = io.BytesIO(plaintext)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        cfg_member = tar.getmember("config.json")
        cfg_data   = tar.extractfile(cfg_member)
        if cfg_data:
            loaded = json.loads(cfg_data.read())
            for k, v in loaded.items():
                if hasattr(settings, k):
                    setattr(settings, k, v)
            settings.save()

        mem_parent = Path(settings.memory_dir).parent
        for member in tar.getmembers():
            if member.name.startswith("memory/"):
                tar.extract(member, path=mem_parent, filter="data")

    log.info("Config imported ← %s", input_path)
