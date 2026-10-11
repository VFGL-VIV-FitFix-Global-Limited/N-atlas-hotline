from app.models.schemas import RetrievalResult
from app.pipeline.rag import RAGRetriever


async def test_retrieve_without_index_is_out_of_knowledge() -> None:
    rag = RAGRetriever.__new__(RAGRetriever)
    rag._threshold = 0.35
    rag._db = None
    rag._index = None
    result = await rag.retrieve("How do I register for NIN?", "government")
    assert isinstance(result, RetrievalResult)
    assert result.is_out_of_knowledge is True
    assert result.chunks == []
