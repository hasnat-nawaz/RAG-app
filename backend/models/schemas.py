import bootstrap
from pathlib import Path
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DEFAULT_TOP_K = 10
MAX_EMBED_BATCH_SIZE = 90

class QueryInput(BaseModel):
    query: NonEmptyStr

class QueryRequest(BaseModel):
    query: NonEmptyStr
    hybrid: bool = False
    hyde: bool = False
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)

    @model_validator(mode='after')
    def at_least_one_method(self):
        if not self.hybrid and (not self.hyde):
            raise ValueError('Select at least one retrieval method: hybrid and/or hyde.')
        return self

class QueryResponse(BaseModel):
    answer: NonEmptyStr
    methods: list[str] = Field(default_factory=list)
    documents_retrieved: int = Field(ge=0)
    documents_used: int = Field(ge=0)

class HybridSearchInput(BaseModel):
    query: NonEmptyStr
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)

class KeywordSearchInput(BaseModel):
    query: NonEmptyStr
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)

class VectorSearchInput(BaseModel):
    query_vector: list[float] = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)

class DocumentLoadInput(BaseModel):
    file_path: Path

    @model_validator(mode='after')
    def file_must_exist(self):
        if not self.file_path.is_file():
            raise ValueError(f'Document not found: {self.file_path}')
        return self

class ChunkMarkdownInput(BaseModel):
    text: NonEmptyStr
    source: NonEmptyStr

class Chunk(BaseModel):
    source: NonEmptyStr
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: NonEmptyStr

class EmbeddableChunk(Chunk):
    embedding_text: NonEmptyStr

class EmbedChunksInput(BaseModel):
    chunks: list[EmbeddableChunk] = Field(min_length=1)
    batch_size: int = Field(default=MAX_EMBED_BATCH_SIZE, ge=1, le=MAX_EMBED_BATCH_SIZE)

class VectorRecord(Chunk):
    id: int = Field(ge=1)
    vector: list[float] = Field(min_length=1)

class AddRecordsInput(BaseModel):
    chunks: list[Chunk] = Field(min_length=1)
    embeddings: list[list[float]] = Field(min_length=1)

    @model_validator(mode='after')
    def lengths_must_match(self):
        if len(self.chunks) != len(self.embeddings):
            raise ValueError(f'Chunk count ({len(self.chunks)}) does not match embedding count ({len(self.embeddings)}).')
        if any((not embedding for embedding in self.embeddings)):
            raise ValueError('Every embedding must be a non-empty list of floats.')
        return self

class RetrievedDocument(BaseModel):
    model_config = ConfigDict(extra='allow', populate_by_name=True)
    id: int | None = None
    source: NonEmptyStr
    content: NonEmptyStr
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] | None = None
    distance: float | None = Field(default=None, alias='_distance')
    score: float | None = Field(default=None, alias='_score')
    rerank_score: float | None = None

class RerankInput(BaseModel):
    query: NonEmptyStr
    documents: list[RetrievedDocument]
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)

class GenerationInput(BaseModel):
    query: NonEmptyStr
    documents: list[RetrievedDocument] = Field(min_length=1)

class GenerationOutput(BaseModel):
    answer: NonEmptyStr

class OptimizedQuery(BaseModel):
    original: NonEmptyStr
    optimized: NonEmptyStr

class HypotheticalDocument(BaseModel):
    query: NonEmptyStr
    document: NonEmptyStr
