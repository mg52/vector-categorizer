# Vector Categorizer

A lightweight HTTP service that classifies text into predefined categories using vector embeddings and cosine similarity.

At startup, the service reads `categories.txt`, embeds each category description via Ollama, and stores the resulting vectors. Incoming text is embedded on demand and matched against those stored vectors — the closest category wins.

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

## Local Setup

### 1. Start Ollama

The service needs an Ollama instance running with `nomic-embed-text` available.

**Option A — Docker (recommended):**

```bash
docker run -d --name ollama -p 11434:11434 ollama/ollama
docker exec ollama ollama pull nomic-embed-text
```

**Option B — Local Ollama install:**

```bash
ollama pull nomic-embed-text
```

Verify it's working:

```bash
curl http://localhost:11434/api/tags
```

### 2. Run the Service

```bash
go run ./cmd
```

The service starts on `http://localhost:8090` by default and logs the loaded categories.

### 3. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VECTOR_INDEX_ADDR` | `:8090` | Listen address |
| `VECTOR_INDEX_EMBED_URL` | `http://localhost:11434/api/embed` | Ollama embed endpoint |
| `VECTOR_INDEX_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `VECTOR_INDEX_CATEGORIES_FILE` | `categories.txt` | Path to categories file |

Example with overrides:

```bash
VECTOR_INDEX_ADDR=:9000 \
VECTOR_INDEX_EMBED_URL=http://my-ollama:11434/api/embed \
go run ./cmd
```

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

Also accepts a plain text body (no `Content-Type` header needed):

```bash
curl -s -X POST http://localhost:8090/categorize \
  --data "planning a trip to Paris next summer"
```

### List Categories

```bash
curl -s http://localhost:8090/categories
```

Response:

```json
{
  "categories": ["food-recipe", "travel", "sports", "technology", "health", "fashion", "music", "movies", "business", "education"]
}
```

### Health Check

```bash
curl -s http://localhost:8090/health
```

## Run Tests

Unit tests use a fake embedder — no Ollama needed:

```bash
go test ./...
```

With verbose output:

```bash
go test ./... -v
```

## Build

```bash
go build -o vector-categorizer ./cmd
./vector-categorizer
```

## Docker Build

```bash
docker build -t vector-categorizer .
docker run --rm -p 8090:8090 \
  -e VECTOR_INDEX_EMBED_URL=http://host.docker.internal:11434/api/embed \
  vector-categorizer
```
