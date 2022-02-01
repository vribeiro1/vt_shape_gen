from tgt import Interval

from vt_shape_gen import get_repeated_phoneme


def test_get_repeated_phoneme():
    phone = Interval(text="a", start_time=0, end_time=1)
    framerate = 55
    repeated_phoneme = get_repeated_phoneme(phone, framerate)

    assert repeated_phoneme == [phone.text] * framerate
