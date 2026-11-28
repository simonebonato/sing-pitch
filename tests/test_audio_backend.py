"""One rule on every OS (docs/cross-platform.md): the app uses the first host API of audio_backend.PREFERRED the
machine has and lists only its devices; WASAPI streams open with auto_convert. Stand-ins for sounddevice's lists of a
Mac (Core Audio only) and of a Windows laptop (four host APIs, MME the default), as measured on real machines."""
import sys
import types

import pytest

from sing_pitch import audio_backend

MAC_APIS = [dict(name="Core Audio", default_input_device=0, default_output_device=1)]
WINDOWS_APIS = [dict(name="MME", default_input_device=1, default_output_device=3),
                dict(name="Windows DirectSound", default_input_device=5, default_output_device=7),
                dict(name="Windows WASAPI", default_input_device=12, default_output_device=10),
                dict(name="Windows WDM-KS", default_input_device=-1, default_output_device=-1)]
WINDOWS_DEVICES = [dict(index=1, name="Microphone (Realtek(R) Audio)", hostapi=0, max_input_channels=2,
                        max_output_channels=0),
                   dict(index=3, name="Headphones (Realtek(R) Audio)", hostapi=0, max_input_channels=0,
                        max_output_channels=2),
                   dict(index=10, name="Headphones (Realtek(R) Audio)", hostapi=2, max_input_channels=0,
                        max_output_channels=2),
                   dict(index=12, name="Microphone (Realtek(R) Audio)", hostapi=2, max_input_channels=2,
                        max_output_channels=0)]


class FakeSd(types.SimpleNamespace):
    def __init__(self, apis, devices=(), default_hostapi=0):
        super().__init__(default=types.SimpleNamespace(hostapi=default_hostapi, device=(None, None)))
        self._apis, self._devices = apis, list(devices)

    def query_hostapis(self, index=None):
        return self._apis if index is None else self._apis[index]

    def query_devices(self, device=None, kind=None):
        return self._devices if device is None else next(d for d in self._devices if d["index"] == device)

    class WasapiSettings:
        def __init__(self, auto_convert=False):
            self.auto_convert = auto_convert


@pytest.fixture
def fake_sd(monkeypatch):
    def install(apis, devices=(), default_hostapi=0):
        sd = FakeSd(apis, devices, default_hostapi)
        monkeypatch.setitem(sys.modules, "sounddevice", sd)
        monkeypatch.setattr(audio_backend, "_done", False)
        monkeypatch.setattr(audio_backend, "_chosen", None)
        return sd
    return install


def test_on_a_mac_core_audio_is_chosen_and_nothing_changes(fake_sd):
    sd = fake_sd(MAC_APIS, [dict(index=0, name="MacBook Pro Microphone", hostapi=0)])
    assert audio_backend.host_api() == 0
    assert sd.default.device == (None, None)                 # Core Audio already is PortAudio's default
    assert len(audio_backend.devices_of_host_api(sd.query_devices())) == 1
    assert audio_backend.stream_options() == {} and audio_backend.stream_options(0) == {}


def test_on_windows_wasapi_is_chosen_its_devices_listed_and_its_defaults_used(fake_sd):
    sd = fake_sd(WINDOWS_APIS, WINDOWS_DEVICES)
    assert audio_backend.host_api() == 2
    assert sd.default.device == (12, 10)
    assert [d["index"] for d in audio_backend.devices_of_host_api(sd.query_devices())] == [10, 12]


def test_wasapi_streams_open_with_auto_convert_and_others_as_before(fake_sd):
    fake_sd(WINDOWS_APIS, WINDOWS_DEVICES)
    assert audio_backend.stream_options()["extra_settings"].auto_convert       # the default device: WASAPI
    assert audio_backend.stream_options(12)["extra_settings"].auto_convert
    assert audio_backend.stream_options(1) == {}                               # an MME device chosen by index


def test_without_a_preferred_host_api_everything_is_left_to_portaudio(fake_sd):
    sd = fake_sd([dict(name="JACK Audio Connection Kit", default_input_device=0, default_output_device=0)],
                 [dict(index=0, name="system", hostapi=0)])
    assert audio_backend.host_api() is None
    assert audio_backend.devices_of_host_api(sd.query_devices()) == sd.query_devices()
    assert audio_backend.stream_options() == {}
