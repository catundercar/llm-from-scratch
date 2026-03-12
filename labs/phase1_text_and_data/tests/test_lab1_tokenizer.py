"""Tests for Phase 1, Lab 1: BPE Tokenizer."""

import pytest
from phase_1.tokenizer import BPETokenizer


SAMPLE_TEXT = "the cat sat on the mat the cat sat"


@pytest.fixture
def trained_tokenizer():
    tok = BPETokenizer()
    tok.train(SAMPLE_TEXT, vocab_size=20)
    return tok


class TestBuildInitialVocab:
    def test_unique_chars(self):
        tok = BPETokenizer()
        ids = tok._build_initial_vocab("abca")
        assert len(tok.vocab) == 3  # a, b, c
        assert len(ids) == 4

    def test_deterministic_ordering(self):
        tok = BPETokenizer()
        tok._build_initial_vocab("cab")
        chars = [tok.vocab[i] for i in sorted(tok.vocab)]
        assert chars == ["a", "b", "c"]


class TestGetPairCounts:
    def test_basic_pairs(self):
        tok = BPETokenizer()
        tok._build_initial_vocab("aab")
        ids = [tok.token_to_id["a"], tok.token_to_id["a"], tok.token_to_id["b"]]
        counts = tok._get_pair_counts(ids)
        a_id, b_id = tok.token_to_id["a"], tok.token_to_id["b"]
        assert counts[(a_id, a_id)] == 1
        assert counts[(a_id, b_id)] == 1

    def test_empty_or_single(self):
        tok = BPETokenizer()
        tok._build_initial_vocab("a")
        assert len(tok._get_pair_counts([0])) == 0
        assert len(tok._get_pair_counts([])) == 0


class TestMergePair:
    def test_basic_merge(self):
        tok = BPETokenizer()
        ids = [0, 1, 0, 1, 2]
        result = tok._merge_pair(ids, (0, 1), 3)
        assert result == [3, 3, 2]

    def test_no_merge(self):
        tok = BPETokenizer()
        ids = [0, 2, 1]
        result = tok._merge_pair(ids, (0, 1), 3)
        assert result == [0, 2, 1]


class TestTrain:
    def test_vocab_size(self, trained_tokenizer):
        assert len(trained_tokenizer.vocab) == 20

    def test_merges_recorded(self, trained_tokenizer):
        assert len(trained_tokenizer.merges) > 0

    def test_merges_grow_vocab(self):
        tok = BPETokenizer()
        tok.train("aaa", vocab_size=3)
        # 1 unique char + merges up to size 3
        assert len(tok.vocab) <= 3


class TestEncodeDecode:
    def test_roundtrip(self, trained_tokenizer):
        text = "the cat sat"
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text

    def test_encode_returns_ids(self, trained_tokenizer):
        ids = trained_tokenizer.encode("the")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_shorter_after_merges(self, trained_tokenizer):
        char_count = len("the cat")
        ids = trained_tokenizer.encode("the cat")
        assert len(ids) <= char_count


class TestSaveLoad:
    def test_save_load_roundtrip(self, trained_tokenizer, tmp_path):
        path = tmp_path / "tok.json"
        trained_tokenizer.save(path)

        tok2 = BPETokenizer()
        tok2.load(path)

        text = "the cat"
        assert trained_tokenizer.encode(text) == tok2.encode(text)
        assert trained_tokenizer.decode(tok2.encode(text)) == text
