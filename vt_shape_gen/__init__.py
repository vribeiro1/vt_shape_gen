import funcy
import os
import torch
import ujson

from functools import lru_cache
from torch.utils.data import DataLoader

from .helpers import set_seeds
from .model import ArtSpeech

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
VOCABULARY_FILEPATH = os.path.join(RESOURCES_DIR, "vocabulary.json")
STATE_DICT_FILEPATH = os.path.join(RESOURCES_DIR, "artspeech_pretrained_weights.pt")


@lru_cache()
def load_tokens():
    """
    Loads the tokens used by the model.

    Return:
    (list): Sorted list of tokens.
    """
    with open(VOCABULARY_FILEPATH) as f:
        tokens = ujson.load(f)

    return sorted(tokens)


@lru_cache()
def load_model(device, state_dict_fpath=None):
    """
    Load the pretrained ArtSpeech model to memory.

    Args:
    device (str): Device type to map the weights. Should be a device supported by PyTorch.

    Return:
    (ArtSpeech): ArtSpeech model with loaded weights.
    """
    device = torch.device(device)
    if state_dict_fpath is None:
        state_dict_fpath = STATE_DICT_FILEPATH

    tokens = load_tokens()
    vocabulary = {token: i for i, token in enumerate(tokens)}

    model = ArtSpeech(len(vocabulary), 11)
    state_dict = torch.load(state_dict_fpath, map_location=device)
    model.load_state_dict(state_dict)

    return model


def get_repeated_token(interval, framerate):
    """
    Creates a list by repeating the token such that the list length matches the token duration and
    the desired framerate.

    Args:
    interval (tgt.Interval): Phoneme to repeat.
    framerate (int): Framerate for generating the vocal tract shape.

    Return:
    (list): List with the repeated token
    """
    period = 1 / framerate
    phone_duration = interval.end_time - interval.start_time

    return [interval.text] * int("%.0f" % (phone_duration / period))


def validate_textgrid(textgrid):
    """
    Validates that the textgrid file has the required tiers (PhonTier and SentenceTier).

    Args:
    textgrid (tgt.core.Textgrid): Textgrid object.

    Return:
    (bool, list): Textgrid is valid, missing tiers.
    """
    textgrid_tier_names = [tier.name for tier in textgrid.tiers]

    required_tiers = ["PhonTier", "SentenceTier"]
    missing_tiers = funcy.lfilter(lambda tier: tier not in textgrid_tier_names, required_tiers)
    is_valid = not any(missing_tiers)

    return is_valid, missing_tiers


def get_phonetic_sequences(textgrid, framerate=55):
    """
    Generates the phonetic sequence for a textgrid object. Token duration in the PhonTier is encoded
    as token repetitions in the output list

    Args:
    textgrid (tgt.core.Textgrid): Textgrid containing at least two tiers: PhonTier and SentenceTier.
    framerate (int): Framerate for vocal tract generation.

    Return:
    (list): List with the tokens for each sentence in the SentenceTier
    """
    textgrid_is_valid, missing_tiers = validate_textgrid(textgrid)
    assert textgrid_is_valid, f"Missing tiers '{missing_tiers}'"

    phone_tier = textgrid.get_tier_by_name("PhonTier")
    sentence_tier = textgrid.get_tier_by_name("SentenceTier")

    sentences = []
    for sentence_interval in sentence_tier:
        sentence_start = sentence_interval.start_time
        sentence_end = sentence_interval.end_time

        in_interval = lambda interval: (
            interval.start_time >= sentence_start and
            interval.end_time <= sentence_end
        )
        phonemes = funcy.lfilter(in_interval, phone_tier)
        repeat_phoneme = funcy.partial(get_repeated_token, framerate=framerate)
        phonetic_sequence = funcy.lflatten(map(repeat_phoneme, phonemes))

        sentences.append(phonetic_sequence)

    return sentences


def generate_vocal_tract_shape(phonetic_sequences, model=None, state_dict_fpath=None, tokens=None):
    """
    Generates the shape of the vocal tract for a sequence of phonemes.

    Args:
    phonetic_sequences (list): Sequence of phonemes with phoneme duration encoded as token
    repetitions.
    model (ArtSpeech): Previously loaded model.
    state_dict_fpath (str): Path to state_dict.
    tokens (str or list): Path to or previously loaded list of tokens.

    Return:
    (list): List containing the predicted shapes for each phonetic sequence. Each array in the
    sequence will have the shape (seq_len, 11, 2, 50).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokens is None:
        tokens = load_tokens()
    vocabulary = {token: i for i, token in enumerate(tokens)}

    sentences_numerized = [
        torch.tensor([vocabulary[token] for token in sentence_tokens], dtype=torch.long)
        for sentence_tokens in phonetic_sequences
    ]

    # Generalize for using batches larger than one
    dataloader = DataLoader(
        list(zip(sentences_numerized, phonetic_sequences)),
        batch_size=1,
        shuffle=False,
        worker_init_fn=set_seeds
    )

    if model is None:
        model = load_model(device.type, state_dict_fpath)
        model.to(device)

    vt_shapes = []
    for inputs, __ in dataloader:
        lengths = [len(sentence) for sentence in inputs]
        inputs = inputs.to(device)  # (bs, seq_len)
        outputs = model(inputs, lengths)  # (bs, seq_len, n_articulators, 2, n_samples)

        outputs = outputs.detach().cpu().numpy()
        vt_shapes += [output for output in outputs]

    return vt_shapes
