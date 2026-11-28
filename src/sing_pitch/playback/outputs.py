"""Is the output device headphones? CoreAudio transport type + data source, with a name heuristic as fallback.
The rule (classify_output) is pure; only output_kind() touches the machine."""
from __future__ import annotations

import re
from typing import Literal

OutputKind = Literal["headphones", "speakers", "unknown"]

_HEADPHONE_NAME = re.compile(r"headphone|airpods|earbud|buds|earphone|headset|wh-|wf-|beats", re.I)
_SPEAKER_NAME = re.compile(r"speaker|soundbar|homepod|sonos", re.I)


def classify_output(name: str, facts: dict | None) -> OutputKind:
    """facts = {"transport": fourcc, "out_source": fourcc|None} from CoreAudio, or None when unavailable."""
    if _HEADPHONE_NAME.search(name):
        return "headphones"
    transport = (facts or {}).get("transport")
    if transport in ("blue", "blea"):
        return "speakers" if _SPEAKER_NAME.search(name) else "headphones"
    if transport == "bltn":
        return "headphones" if facts.get("out_source") == "hdpn" else "speakers"
    if transport in ("hdmi", "dprt"):
        return "speakers"
    if transport is None and _SPEAKER_NAME.search(name):
        return "speakers"
    return "unknown"          # USB, aggregate, virtual, or nothing known: warn softly


def output_kind(name: str) -> OutputKind:
    """Classify a sounddevice output name on this Mac. Never raises."""
    try:
        from .coreaudio import info

        facts = info().get(name)
    except Exception:
        facts = None
    return classify_output(name, facts)


def output_devices(devices) -> list[dict]:
    """The output-capable ones of sounddevice's device dicts. Pure, so the window is tested with fake lists."""
    return [dict(d) for d in devices if d["max_output_channels"] > 0]


def list_outputs() -> list[dict]:
    import sounddevice as sd

    from ..audio_backend import devices_of_host_api

    return output_devices(devices_of_host_api(sd.query_devices()))


def default_output_name() -> str | None:
    try:
        import sounddevice as sd

        from ..audio_backend import host_api

        host_api()                        # the default device of the chosen host API
        return sd.query_devices(kind="output")["name"]
    except Exception:
        return None


def headphone_warning(kind: OutputKind, name: str | None) -> str | None:
    """Text for the window, or None when the output is headphones."""
    if kind == "headphones":
        return None
    shown = name or "the output device"
    if kind == "speakers":
        return f"Output is {shown}, which looks like speakers: the Backing will leak into the mic. Use headphones."
    return f"Cannot tell whether {shown} is headphones. If it is speakers, the Backing will leak into the mic."
