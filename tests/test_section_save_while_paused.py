"""#50 round 2: the tester saw "song position moved while paused" after saving or choosing a Section. Saving a Section
that is looping only names it: the song, paused or not, stays where it is and the Pass goes on. Choosing another
Section while looping moves the song on purpose (the new Section's Lead-in)."""
from app_speed import APP_SPEED
from pause_fuzz import build_song

from sing_pitch.qa import QA


def test_saving_the_looping_unsaved_mark_does_not_move_the_paused_song_or_restart_the_pass(tmp_path):
    build_song(tmp_path)
    with QA(tmp_path, mic="echo", time_scale=APP_SPEED) as qa:
        qa.set("reference_box", "Melody")
        qa.mark_section(10.0, 14.0)
        qa.set("loop_box", True)
        qa.start_take(at_s=11)
        qa.wait(1.0)
        qa.pause()
        qa.wait(0.4)
        where, number = qa.song_position(), qa.window._backing.pass_index
        qa.set("section_name_edit", "Hook")
        qa.click("save_section_button")
        qa.wait(0.5)
        assert qa.window.section_box.currentText() == "Hook"
        assert qa.song_position() == where and qa.window._backing.pass_index == number
        assert qa.window._backing.loop.name == "Hook" and qa.window._paused
        qa.pause()
        qa.wait(0.3)
        assert qa.song_position() > where
        qa.stop_take()


def test_choosing_another_section_while_looping_moves_the_song_to_its_lead_in_on_purpose(tmp_path):
    build_song(tmp_path)
    with QA(tmp_path, mic="echo", time_scale=APP_SPEED) as qa:
        qa.set("reference_box", "Melody")
        qa.mark_section(10.0, 14.0)
        qa.set("section_name_edit", "Hook")
        qa.click("save_section_button")
        qa.mark_section(20.0, 24.0)
        qa.set("loop_box", True)
        qa.start_take(at_s=21)
        qa.wait(0.5)
        qa.pause()
        qa.wait(0.3)
        qa.set("section_box", "Hook")
        qa.wait(0.4)
        assert 7.9 < qa.song_position() < 8.5                  # Hook starts at 10 s, its Lead-in 2 s before
        assert qa.window._paused
        qa.stop_take()
