import pytest

from dental.llm import NoProviderConfigured, resolved_provider


@pytest.fixture(autouse=True)
def clear_keys(monkeypatch):
    for key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_no_keys_configured():
    from dental.llm import get_llm

    assert resolved_provider() == "none"
    with pytest.raises(NoProviderConfigured) as excinfo:
        get_llm()
    message = str(excinfo.value)
    for key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        assert key in message


def test_groq_wins_when_multiple_keys_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_API_KEY", "r")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert resolved_provider() == "groq"


def test_falls_back_through_cascade(monkeypatch):
    """Cascade order is fixed: an earlier provider wins even when a later one is set."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "r")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert resolved_provider() == "openrouter"
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert resolved_provider() == "gemini"
    monkeypatch.setenv("GROQ_API_KEY", "g")
    assert resolved_provider() == "groq"


def test_removing_the_winning_key_falls_through(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_API_KEY", "r")
    assert resolved_provider() == "gemini"
    monkeypatch.delenv("GOOGLE_API_KEY")
    assert resolved_provider() == "openrouter"


def test_openai_used_when_it_is_the_only_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert resolved_provider() == "openai"
