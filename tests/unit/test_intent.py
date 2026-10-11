from app.models.schemas import Domain
from app.pipeline.intent import IntentClassifier


def test_english_government(intent_classifier: IntentClassifier) -> None:
    assert intent_classifier.classify("How do I register for NIN?", "en") == Domain.GOVERNMENT


def test_english_health(intent_classifier: IntentClassifier) -> None:
    assert intent_classifier.classify("My child has fever and malaria", "en") == Domain.HEALTH


def test_english_market(intent_classifier: IntentClassifier) -> None:
    assert intent_classifier.classify("How much is a bag of rice at the market?", "en") == Domain.MARKET


def test_english_agriculture(intent_classifier: IntentClassifier) -> None:
    assert intent_classifier.classify("When should I plant cassava?", "en") == Domain.AGRICULTURE


def test_yoruba_health(intent_classifier: IntentClassifier) -> None:
    assert intent_classifier.classify("Mo ní ibà", "yo") == Domain.HEALTH


def test_hausa_market(intent_classifier: IntentClassifier) -> None:
    assert intent_classifier.classify("Nawa ne a kasuwa?", "ha") == Domain.MARKET


def test_igbo_agriculture(intent_classifier: IntentClassifier) -> None:
    assert intent_classifier.classify("Kedu ka m ga-esi kụọ osikapa n'ubi?", "ig") == Domain.AGRICULTURE


def test_code_switching(intent_classifier: IntentClassifier) -> None:
    assert intent_classifier.classify("Mo need NIN registration", "yo") == Domain.GOVERNMENT


def test_unknown_returns_general(intent_classifier: IntentClassifier) -> None:
    assert intent_classifier.classify("tell me a joke", "en") == Domain.GENERAL


def test_empty_returns_general(intent_classifier: IntentClassifier) -> None:
    assert intent_classifier.classify("", "en") == Domain.GENERAL
    assert intent_classifier.classify("   ", "en") == Domain.GENERAL


def test_highest_scoring_domain(intent_classifier: IntentClassifier) -> None:
    text = "How much does fertilizer cost at the market for my farm crop?"
    result = intent_classifier.classify(text, "en")
    assert result in {Domain.MARKET, Domain.AGRICULTURE}
