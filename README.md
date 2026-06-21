# Vector Categorizer

A lightweight HTTP service that classifies text into predefined categories using vector embeddings and cosine similarity.

At startup, the service reads `categories.txt`, embeds each category description via the configured provider, and stores the resulting vectors. Incoming text is embedded on demand and matched against those stored vectors — the closest category wins.

## How It Works

1. On startup: each category description in `categories.txt` is embedded and normalized.
2. On request: the query text is embedded and normalized.
3. Cosine similarity is computed against every category vector.
4. The category with the highest similarity is returned, along with its similarity score and distance (`1 - similarity`).

## categories.txt Format

```
# Lines starting with # are ignored
food-recipe: food, recipe, cooking, meal, lunch, dinner, ingredients
travel: trip, journey, destination, city, vacation, hotel, flight
```

Each line: `category-name: description text used for embedding`

Edit this file to add, remove, or tune categories. The service must be restarted to pick up changes.

## Embedding Providers

The service supports multiple embedding providers via `VECTOR_INDEX_EMBED_PROVIDER`.

### Ollama (default)

Requires a running Ollama instance with `nomic-embed-text` pulled.

**Docker:**
```bash
docker run -d --name ollama -p 11434:11434 ollama/ollama
docker exec ollama ollama pull nomic-embed-text
```

**Local:**
```bash
ollama pull nomic-embed-text
```

```bash
go run ./cmd
```

### OpenAI

```bash
VECTOR_INDEX_EMBED_PROVIDER=openai \
OPENAI_API_KEY=sk-... \
go run ./cmd
```

Default model is `text-embedding-3-small`. To use a different model:

```bash
VECTOR_INDEX_EMBED_PROVIDER=openai \
OPENAI_API_KEY=sk-... \
VECTOR_INDEX_EMBED_MODEL=text-embedding-3-large \
go run ./cmd
```

### Adding a Custom Provider

Implement the `Embedder` interface in `internal/vectorindex` and wire it in `cmd/main.go`:

```go
type Embedder interface {
    Embed(ctx context.Context, text string) ([]float64, error)
}
```

Optionally also implement `BatchEmbedder` for more efficient bulk embedding at startup:

```go
type BatchEmbedder interface {
    EmbedBatch(ctx context.Context, texts []string) ([][]float64, error)
}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VECTOR_INDEX_ADDR` | `:8090` | Listen address |
| `VECTOR_INDEX_EMBED_PROVIDER` | `ollama` | Embedding provider (`ollama` or `openai`) |
| `VECTOR_INDEX_EMBED_URL` | `http://localhost:11434/api/embed` | Ollama embed endpoint |
| `VECTOR_INDEX_EMBED_MODEL` | `nomic-embed-text` / `text-embedding-3-small` | Embedding model (default depends on provider) |
| `VECTOR_INDEX_CATEGORIES_FILE` | `categories.txt` | Path to categories file |
| `OPENAI_API_KEY` | — | Required when provider is `openai` |

## Endpoints

### Categorize Text

```bash
curl -s -X POST http://localhost:8090/categorize \
  -H "Content-Type: application/json" \
  -d '{"text": "I want to cook pasta with tomato sauce for dinner"}'
```

Response:

```json
{
  "category": "food-recipe",
  "similarity": 0.5457,
  "distance": 0.4543
}
```

Also accepts a plain text body:

```bash
curl -s -X POST http://localhost:8090/categorize \
  --data "planning a trip to Paris next summer"
```

### List Categories

```bash
curl -s http://localhost:8090/categories
```

### Health Check

```bash
curl -s http://localhost:8090/health
```

## Run Tests

Unit tests use a fake embedder — no external dependency needed:

```bash
go test ./...
```

## Build

```bash
go build -o vector-categorizer ./cmd
./vector-categorizer
```

## Docker

```bash
docker build -t vector-categorizer .

# with Ollama
docker run --rm -p 8090:8090 \
  -e VECTOR_INDEX_EMBED_URL=http://host.docker.internal:11434/api/embed \
  vector-categorizer

# with OpenAI
docker run --rm -p 8090:8090 \
  -e VECTOR_INDEX_EMBED_PROVIDER=openai \
  -e OPENAI_API_KEY=sk-... \
  vector-categorizer
```
