import pytest
from app.services.store import ensure_db, add_text
from app.services.search import search_similar

def test_search_runs_without_docs():
    ensure_db()
    results = search_similar("test query", top_k=3)
    assert isinstance(results, list)
