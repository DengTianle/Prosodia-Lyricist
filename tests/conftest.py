import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BartConfig, BartForConditionalGeneration, PreTrainedTokenizerFast

from prosodia_lyricist.features import configure_tokenizer
from prosodia_lyricist.legacy_model import ProsodyBart


@pytest.fixture
def tokenizer():
    words = ["<s>", "<pad>", "</s>", "<unk>", "hello", "world", "a", "song", "light", "."]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    return configure_tokenizer(tokenizer, 8)


@pytest.fixture
def tiny_model(tokenizer):
    config = BartConfig(
        vocab_size=len(tokenizer),
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=64,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
    )
    return ProsodyBart(BartForConditionalGeneration(config), max_syllables=8, dropout=0.0)


@pytest.fixture
def annotation():
    return {
        "info": {
            "id": "song-a",
            "title": "a song",
            "artist": "Artist",
            "metadata": {"language": "english"},
            "scores": {"NCC": 0.9},
        },
        "annotations": {
            "type": "horizontal",
            "annot": {
                "lines": [{"text": "hello world", "time": [0, 4], "index": 0}],
                "words": [
                    {"text": "hello", "time": [0, 2], "index": 0},
                    {"text": "world", "time": [2, 4], "index": 0},
                ],
                "notes": [
                    {"text": "hel", "time": [0, 0.5], "index": 0},
                    {"text": "lo", "time": [0.5, 1], "index": 0},
                    {"text": "~", "time": [1.5, 2], "index": 0},
                    {"text": "world", "time": [2, 4], "index": 1},
                ],
            },
        },
    }


@pytest.fixture
def record():
    return {
        "title": "a song",
        "text": "hello world",
        "syllables": [
            {"stress": "weak", "length": "short"},
            {"stress": "strong", "length": "long"},
            {"stress": "strong", "length": "short"},
        ],
    }
