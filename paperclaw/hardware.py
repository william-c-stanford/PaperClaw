"""Hardware / environment detection for PaperClaw.

Detects the compute resources available for running *real* experiments —
CPU / GPU / memory / disk — for one or more machines. Two scopes:

  * LOCAL — the machine this process runs on. In CLI *local* mode that is the
    user's own computer; in *backend* mode (CLI ``--backend`` or the web
    frontend talking to a deployed backend) "local" is the host running the
    backend. Detection runs wherever this code executes, so it is correct for
    both without special-casing.

    NOTE / PLACEHOLDER — the meaning of "local" under a future hosted /
    subscription model is undecided. If PaperClaw is ever offered as a
    managed service, "local" should point at the *user's* computer (e.g. via a
    local agent), NOT the shared backend host. Until that product decision is
    made we treat local = the host running this process, which is safe today
    because the user deploys their own backend.

  * REMOTE — a machine reached over SSH (key-based; no password is stored). SSH
    targets are configured in Settings.

The same probe script runs locally and over SSH, so parsing is identical.
Linux-first: fields that depend on ``lscpu`` / ``lsblk`` / ``nvidia-smi`` fall
back to best-effort or blank on other platforms.
"""

import platform
import re
import subprocess
import time

from paperclaw.server.models import DiskInfo, GpuInfo, HardwareInfo, SSHTarget

# One portable-ish probe. Each section is delimited by a ``##NAME##`` marker so
# the same parser handles local stdout and SSH stdout.
_PROBE = r"""
echo "##OS##"
uname -srm 2>/dev/null || true
echo "##CPU_MODEL##"
if command -v lscpu >/dev/null 2>&1; then lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -n1; fi
echo "##CPU_CORES##"
if command -v lscpu >/dev/null 2>&1; then lscpu 2>/dev/null | sed -n 's/^Core(s) per socket:[[:space:]]*//p' | head -n1; fi
echo "##CPU_SOCKETS##"
if command -v lscpu >/dev/null 2>&1; then lscpu 2>/dev/null | sed -n 's/^Socket(s):[[:space:]]*//p' | head -n1; fi
echo "##CPU_THREADS##"
nproc 2>/dev/null || true
echo "##MEM_KB##"
sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\).*/\1/p' /proc/meminfo 2>/dev/null || true
echo "##GPU##"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null; fi
echo "##DISK##"
if command -v lsblk >/dev/null 2>&1; then lsblk -d -b -P -o NAME,MODEL,SIZE,ROTA,TRAN 2>/dev/null; fi
echo "##END##"
"""

_PROBE_TIMEOUT = 25  # seconds — generous for a slow SSH link
_SKIP_DISK_PREFIXES = ("loop", "ram", "zram", "sr", "fd", "dm-")


def _parse_sections(output: str) -> dict[str, list[str]]:
    """Split probe stdout into ``{section: [lines]}`` on ``##NAME##`` markers."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in output.splitlines():
        line = raw.rstrip()
        m = re.fullmatch(r"##(\w+)##", line.strip())
        if m:
            current = m.group(1)
            sections[current] = []
        elif current is not None and line.strip():
            sections[current].append(line)
    return sections


def _first(sections: dict[str, list[str]], key: str) -> str | None:
    vals = sections.get(key) or []
    return vals[0].strip() if vals else None


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None and value.strip() else None
    except ValueError:
        return None


_KV_RE = re.compile(r'(\w+)="([^"]*)"')


def _parse_disks(lines: list[str]) -> list[DiskInfo]:
    disks: list[DiskInfo] = []
    for line in lines:
        kv = dict(_KV_RE.findall(line))
        name = kv.get("NAME", "").strip()
        if not name or name.startswith(_SKIP_DISK_PREFIXES):
            continue
        tran = (kv.get("TRAN") or "").strip().lower() or None
        rota = (kv.get("ROTA") or "").strip()
        if tran == "nvme":
            kind = "NVMe"
        elif rota == "1":
            kind = "HDD"
        elif rota == "0":
            kind = "SSD"
        else:
            kind = "unknown"
        size_b = _int(kv.get("SIZE"))
        disks.append(DiskInfo(
            name=name,
            model=(kv.get("MODEL") or "").strip() or None,
            sizeGb=round(size_b / 1e9, 1) if size_b else None,
            kind=kind,
            transport=tran,
        ))
    return disks


def _parse_gpus(lines: list[str]) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line in lines:
        # "NVIDIA GeForce RTX 3090, 24576"
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        gpus.append(GpuInfo(name=parts[0], memoryTotalMb=_int(parts[1]) if len(parts) > 1 else None))
    return gpus


def _build_info(scope: str, label: str, output: str) -> HardwareInfo:
    sections = _parse_sections(output)

    cores = _int(_first(sections, "CPU_CORES"))
    sockets = _int(_first(sections, "CPU_SOCKETS"))
    physical = cores * sockets if (cores and sockets) else None
    threads = _int(_first(sections, "CPU_THREADS"))
    mem_kb = _int(_first(sections, "MEM_KB"))

    cpu_model = _first(sections, "CPU_MODEL")
    os_str = _first(sections, "OS")
    # Local best-effort fallbacks when lscpu/uname are unavailable (e.g. macOS).
    if scope == "local":
        if not cpu_model:
            cpu_model = platform.processor() or None
        if not os_str:
            os_str = f"{platform.system()} {platform.release()} {platform.machine()}".strip()
        if not threads:
            import os
            threads = os.cpu_count()

    return HardwareInfo(
        scope=scope,  # type: ignore[arg-type]
        label=label,
        reachable=True,
        os=os_str,
        cpuModel=cpu_model,
        cpuCores=physical or threads,
        cpuThreads=threads,
        memTotalGb=round(mem_kb / 1024 / 1024, 1) if mem_kb else None,
        gpus=_parse_gpus(sections.get("GPU", [])),
        disks=_parse_disks(sections.get("DISK", [])),
        detectedAt=time.time(),
    )


def detect_local() -> HardwareInfo:
    """Probe the machine this process runs on."""
    try:
        proc = subprocess.run(
            ["bash", "-c", _PROBE],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT,
        )
        return _build_info("local", "local", proc.stdout)
    except Exception as exc:  # bash missing, timeout, etc.
        return HardwareInfo(
            scope="local", label="local", reachable=False, error=str(exc),
            os=f"{platform.system()} {platform.release()} {platform.machine()}".strip(),
            detectedAt=time.time(),
        )


def detect_remote(target: SSHTarget) -> HardwareInfo:
    """Probe a remote machine over SSH (key-based, non-interactive).

    Pipes the probe script to ``bash -s`` on the remote so no quoting gymnastics
    are needed. Never blocks on a password — ``BatchMode=yes`` fails fast if the
    key isn't accepted.
    """
    label = f"{target.user}@{target.host}"
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
        "-p", str(target.port),
    ]
    if target.key_path:
        cmd += ["-i", target.key_path]
    cmd += [label, "bash -s"]
    try:
        proc = subprocess.run(
            cmd, input=_PROBE, capture_output=True, text=True, timeout=_PROBE_TIMEOUT,
        )
        if proc.returncode != 0 and "##END##" not in proc.stdout:
            err = (proc.stderr or "").strip() or f"ssh exited {proc.returncode}"
            return HardwareInfo(
                scope="remote", label=label, reachable=False, error=err,
                detectedAt=time.time(),
            )
        info = _build_info("remote", label, proc.stdout)
        return info
    except Exception as exc:  # ssh missing, timeout, host unreachable
        return HardwareInfo(
            scope="remote", label=label, reachable=False, error=str(exc),
            detectedAt=time.time(),
        )


# ── HARDWARE.md rendering ─────────────────────────────────────────────────────

def _machine_section(m: HardwareInfo) -> str:
    head = f"## {m.label}" + ("" if m.scope == "local" else " (remote)")
    if not m.reachable:
        return f"{head}\n\n_Unreachable: {m.error or 'unknown error'}_\n"
    lines = [head, ""]
    if m.os:
        lines.append(f"- **OS:** {m.os}")
    if m.cpu_model or m.cpu_threads:
        cpu = m.cpu_model or "Unknown CPU"
        counts = []
        if m.cpu_cores:
            counts.append(f"{m.cpu_cores} cores")
        if m.cpu_threads:
            counts.append(f"{m.cpu_threads} threads")
        suffix = f" ({' / '.join(counts)})" if counts else ""
        lines.append(f"- **CPU:** {cpu}{suffix}")
    if m.mem_total_gb:
        lines.append(f"- **Memory:** {m.mem_total_gb:g} GB")
    if m.gpus:
        gpu_strs = [
            g.name + (f" ({g.memory_total_mb / 1024:.0f} GB)" if g.memory_total_mb else "")
            for g in m.gpus
        ]
        lines.append(f"- **GPU:** {'; '.join(gpu_strs)}")
    else:
        lines.append("- **GPU:** none detected")
    if m.disks:
        lines.append("- **Disks:**")
        lines.append("")
        lines.append("  | Device | Model | Size | Type |")
        lines.append("  |---|---|---|---|")
        for d in m.disks:
            size = f"{d.size_gb:g} GB" if d.size_gb else "?"
            lines.append(f"  | {d.name} | {d.model or '?'} | {size} | {d.kind} |")
    lines.append("")
    return "\n".join(lines)


def render_markdown(machines: list[HardwareInfo]) -> str:
    """Deterministic HARDWARE.md from the detected machines."""
    today = time.strftime("%B %d, %Y")
    out = [
        "# Hardware & Environment",
        "",
        "> Detected compute resources available for running experiments.",
        f"> Updated {today}.",
        "",
    ]
    for m in machines:
        out.append(_machine_section(m))
    return "\n".join(out).rstrip() + "\n"
