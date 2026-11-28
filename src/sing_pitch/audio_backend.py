"""Which PortAudio host API the app uses, and how a stream is opened on it. The one place that knows host APIs differ.

PortAudio lists every device once per host API on the machine: a Mac has one (Core Audio), Windows has four (MME,
DirectSound, WASAPI, WDM-KS) and starts on MME. The rule is the same everywhere: use the first host API of
PREFERRED that the machine has, list only its devices, and make its defaults sounddevice's defaults. On a Mac that
is Core Audio, which is already the only one and the default, so nothing changes there.

Why WASAPI on Windows (measured 2026-11-26, Realtek laptop, docs/cross-platform.md): MME adds ~200 ms of output
latency and stamps input chunks with no host time (currentTime 0, inputBufferAdcTime a few ms within the buffer);
WASAPI measured 46 ms out / 22 ms in with real stamps on PortAudio's clock. WASAPI shared mode only opens a
stream at the device's own rate and channel count, so its auto_convert flag is set: then the app's 48 kHz mono
input and the song's own rate open on any device, as they do on Core Audio."""
from __future__ import annotations

PREFERRED = ("Core Audio", "Windows WASAPI", "ALSA")   # the best of each OS; a machine has at most one of them

_chosen: int | None = None
_done = False


def host_api() -> int | None:
    """Index of the host API the app uses (None: PortAudio has none of PREFERRED, then its own default is used).
    The first call also points sounddevice's default devices at that host API when it is not PortAudio's default."""
    global _chosen, _done
    if _done:
        return _chosen
    try:
        import sounddevice as sd

        names = [h["name"] for h in sd.query_hostapis()]
        chosen = next((names.index(p) for p in PREFERRED if p in names), None)
        if chosen is not None and chosen != sd.default.hostapi:
            api = sd.query_hostapis(chosen)
            sd.default.device = (api["default_input_device"], api["default_output_device"])
    except Exception:            # no PortAudio (or a test's stand-in): PortAudio's own default, asked again next time
        return None
    _chosen, _done = chosen, True
    return _chosen


def devices_of_host_api(devices) -> list:
    """The ones of sounddevice's device dicts on the chosen host API (all of them when none is chosen)."""
    devices = list(devices)
    chosen = host_api()
    return devices if chosen is None else [d for d in devices if d.get("hostapi", chosen) == chosen]


def stream_options(device=None) -> dict:
    """Extra keyword arguments for a sounddevice stream on `device` (None = the default device)."""
    chosen = host_api()
    if chosen is None:
        return {}
    try:
        import sounddevice as sd

        api = device if device is None else sd.query_devices(device)["hostapi"]
        name = sd.query_hostapis(chosen if api is None else api)["name"]
        return {"extra_settings": sd.WasapiSettings(auto_convert=True)} if name == "Windows WASAPI" else {}
    except Exception:
        return {}
