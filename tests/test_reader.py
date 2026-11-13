"""Cerveau lecteur : exercices fabriqués à partir de la bibliothèque."""

import random

import numpy as np

from miradorsi.library import Library, LibraryBuilder
from miradorsi.model.reader import (NOT_FOUND, encode_example, format_prompt, in_phrase, make_examples,
                                    subject_forms, tidy, unsupported)


def _library(tmp_path):
    builder = LibraryBuilder(tmp_path / "b.db", "fr")
    builder.add("Japon", "Le Japon (en japonais : 日本, Nihon) est un pays insulaire d'Asie de l'Est.\n"
                         "Japon — Capitale : Tokyo.\nJapon — Langues officielles : Japonais.", 3e-5)
    builder.add("Australie", "L'Australie est un pays d'Océanie, entre les océans Pacifique et Indien.\n"
                             "Australie — Capitale : Canberra.", 2e-5)
    builder.add("Lyon", "Lyon (prononcé /ljɔ̃/) est une commune française au confluent du Rhône et de la Saône.\n"
                        "Lyon — Population municipale : 519 127 hab. (2023).", 1e-5)
    builder.close()
    return Library(tmp_path / "b.db")


def test_french_forms():
    assert subject_forms("Japon", "Le Japon est un pays") == ("le Japon", "du Japon")
    assert subject_forms("Australie", "L'Australie est") == ("l'Australie", "de l'Australie")
    assert subject_forms("Les Misérables", "Les Misérables est un roman") == ("Les Misérables", "des Misérables")
    assert subject_forms("Lyon", "Lyon est une commune") == ("Lyon", "de Lyon")
    assert in_phrase("de la France", "la France") == "en France" and in_phrase("de Lyon", "Lyon") == "à Lyon"
    assert tidy("Paris (/pa.ʁi/) est la capitale. Le Japon (日本) est un pays.") == \
        "Paris est la capitale. Le Japon est un pays."


def test_exercises_have_traps_citations_and_honest_refusals(tmp_path):
    library = _library(tmp_path)
    japon = library.find("Japon")
    others = [library.find("Australie"), library.find("Lyon")]
    examples = make_examples(library, japon, random.Random(1), others, absent_ratio=0)
    kinds = {ex.kind.split(":")[0] for ex in examples}
    assert kinds == {"fiche", "definition"}
    for ex in examples:
        cited = int(ex.answer.rsplit("[", 1)[1].rstrip("]"))
        assert ex.sources[cited - 1][0] == "Japon"  # la réponse cite la bonne source
        assert "日本" not in format_prompt(ex.question, ex.sources)
    fiche = next(ex for ex in examples if ex.kind.startswith("fiche"))
    assert fiche.answer in ("La capitale du Japon est Tokyo. [{}]".format(fiche.answer[-2]),
                            "Les langues officielles du Japon sont : Japonais. [{}]".format(fiche.answer[-2]))
    absent = make_examples(library, japon, random.Random(1), others, absent_ratio=1.0)
    assert all(ex.answer == NOT_FOUND and all(t != "Japon" for t, _ in ex.sources) for ex in absent)


def test_only_the_answer_is_trained(tmp_path):
    from miradorsi.model.tokenizer import BPETokenizer

    library = _library(tmp_path)
    example = make_examples(library, library.find("Lyon"), random.Random(2),
                            [library.find("Japon"), library.find("Australie")], absent_ratio=0)[0]
    tokenizer = BPETokenizer.train(format_prompt(example.question, example.sources) + example.answer, 300)
    x, y = encode_example(tokenizer, example, 512)
    trained = tokenizer.decode([int(t) for t in y[y >= 0]])
    assert trained == example.answer + "<|end|>" and len(x) == len(y)
    assert encode_example(tokenizer, example, 10) is None  # trop long : écarté


def test_unsupported_facts_are_detected():
    sources = [("Japon", "Le Japon a pour capitale Tokyo depuis 1868.")]
    assert unsupported("La capitale du Japon est Tokyo. [1]", "Capitale du Japon ?", sources) == []
    assert unsupported("La capitale du Japon est Kyoto depuis 794. [1]", "Capitale du Japon ?", sources) == \
        ["Kyoto", "794"]
