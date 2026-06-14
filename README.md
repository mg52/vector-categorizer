# Vector Categorizer

Vector Categorizer is a small HTTP service that turns text documents into embeddings and groups them by category centroids.

It supports two modes:

- **Supervised category mode:** if a document has a `category`, the service groups it under that category and updates that category centroid.
- **Automatic clustering mode:** if a document has no `category`, the service assigns it to the nearest existing centroid when similarity is high enough, otherwise it seeds an empty category slot.

The service uses a configurable embedding API. By default it expects a `/api/embed` response shape with an `embeddings` array and the `nomic-embed-text` model.

## How It Works

1. A document is sent to the service.
2. The service calls the configured embedding API and converts the text into an embedding vector.
3. The vector is normalized.
4. If the request includes `category`, that category's centroid is updated.
5. If the request does not include `category`, the vector is compared with existing category centroids using cosine similarity.
6. The selected category centroid is updated with incremental average:

```text
newCentroid = ((oldCentroid * documentCount) + newVector) / (documentCount + 1)
```

The service does not recalculate old documents when updating a centroid.

## Local Setup

### 1. Start An Embedding Provider

Make sure your embedding endpoint is available at:

```text
http://localhost:11434/api/embed
```

Prepare the embedding model in your provider:

```bash
nomic-embed-text
```

If you are using Docker, expose your embedding provider on port `11434` or override the URL with `VECTOR_INDEX_EMBED_URL`.

### 2. Run The Service

```bash
cd /Users/mustafagordesli/Documents/GoProjects/vector-categorizer
go run ./cmd
```

By default the service starts on:

```text
http://localhost:8090
```

### 3. Optional Configuration

```bash
VECTOR_INDEX_ADDR=:8090 \
VECTOR_INDEX_CATEGORY_COUNT=100 \
VECTOR_INDEX_SIMILARITY_THRESHOLD=0.60 \
VECTOR_INDEX_EMBED_URL=http://localhost:11434/api/embed \
VECTOR_INDEX_EMBED_MODEL=nomic-embed-text \
go run ./cmd
```

## Endpoints

### Health

```bash
curl -s http://localhost:8090/health
```

### Batch Index With Categories

This is the recommended mode when you already know the category labels.

```bash
curl -s http://localhost:8090/index/batch \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": "doc-1", "text": "Wireless Bluetooth headphones", "category": "audio"},
      {"id": "doc-2", "text": "Noise cancelling studio headset", "category": "audio"},
      {"id": "doc-3", "text": "Lightweight waterproof hiking jacket", "category": "outdoor"},
      {"id": "doc-4", "text": "Insulated snowproof winter parka", "category": "outdoor"}
    ]
  }'
```

The response includes the assigned category, category ID, similarity, and current category document count.

### Predict Category

After indexing category-labeled documents, use `/index/search` to predict the closest category for a new text.

```bash
curl -s http://localhost:8090/index/search \
  -H "Content-Type: application/json" \
  -d '{"text":"ear worn sound device for listening to songs"}'
```

Example response:

```json
{
  "categoryID": 0,
  "category": "audio",
  "similarity": 0.73,
  "aboveThreshold": true,
  "categoryDocumentCount": 2
}
```

### Single Index

```bash
curl -s http://localhost:8090/index \
  -H "Content-Type: application/json" \
  -d '{"id":"doc-5","text":"Portable waterproof speaker","category":"audio"}'
```

### Categories

```bash
curl -s "http://localhost:8090/categories"
```

Include centroid vectors:

```bash
curl -s "http://localhost:8090/categories?centroids=true"
```

### Document Category

```bash
curl -s "http://localhost:8090/document-category?id=doc-1"
```

## Run Tests

```bash
cd /Users/mustafagordesli/Documents/GoProjects/vector-categorizer
go test ./...
```

## Build

```bash
cd /Users/mustafagordesli/Documents/GoProjects/vector-categorizer
go build ./cmd
```

This creates a local binary named `cmd` unless you pass `-o`.

Example:

```bash
go build -o vector-categorizer ./cmd
./vector-categorizer
```
