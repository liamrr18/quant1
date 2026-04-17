#!/usr/bin/env python3
"""Helper for running commands on the VPS and uploading files.

Credentials come from environment variables — see .env.example.
"""

import os
import sys
import paramiko
from pathlib import Path

HOST = os.environ.get("VPS_HOST", "")
USER = os.environ.get("VPS_USER", "root")
PASSWORD = os.environ.get("VPS_PASSWORD", "")

if not HOST or not PASSWORD:
    raise RuntimeError("Set VPS_HOST and VPS_PASSWORD in environment (see .env.example)")


def ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    return c


def run(cmd, show=True, check=True):
    """Run a command on the VPS, stream output."""
    c = ssh()
    stdin, stdout, stderr = c.exec_command(cmd, get_pty=False, timeout=600)
    out = stdout.read().decode()
    err = stderr.read().decode()
    rc = stdout.channel.recv_exit_status()
    c.close()
    if show:
        if out:
            try: print(out)
            except UnicodeEncodeError: print(out.encode('ascii', 'replace').decode())
        if err:
            try: print("STDERR:", err, file=sys.stderr)
            except UnicodeEncodeError: print("STDERR:", err.encode('ascii', 'replace').decode(), file=sys.stderr)
    if check and rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}\n{err}")
    return rc, out, err


def upload(local_path, remote_path):
    """Upload a single file to the VPS."""
    c = ssh()
    sftp = c.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()
    c.close()


def upload_dir(local_dir, remote_dir, exclude=None):
    """Upload a directory recursively."""
    exclude = exclude or set()
    c = ssh()
    sftp = c.open_sftp()

    local = Path(local_dir)
    # Ensure remote dir exists
    try:
        sftp.mkdir(remote_dir)
    except IOError:
        pass

    for item in local.rglob("*"):
        rel = item.relative_to(local)
        parts = rel.parts
        if any(p in exclude for p in parts):
            continue
        remote = f"{remote_dir}/{'/'.join(parts)}"
        if item.is_dir():
            try:
                sftp.mkdir(remote)
            except IOError:
                pass
        else:
            sftp.put(str(item), remote)
    sftp.close()
    c.close()


if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:])
    rc, out, err = run(cmd, check=False)
    sys.exit(rc)
