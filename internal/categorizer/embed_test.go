package categorizer

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNewEmbedClientUsesDefaults(t *testing.T) {
	client := NewEmbedClient("", "")
	if client.URL != DefaultEmbedURL {
		t.Fatalf("default URL mismatch: got=%s", client.URL)
	}
	if client.Model != DefaultEmbedModel {
		t.Fatalf("default model mismatch: got=%s", client.Model)
	}
	if client.Client == nil {
		t.Fatalf("expected default HTTP client")
	}
}

func TestEmbedClientEmbedBatchPostsExpectedRequest(t *testing.T) {
	var gotRequest embedRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method mismatch: got=%s", r.Method)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Fatalf("content type mismatch: got=%s", r.Header.Get("Content-Type"))
		}
		if err := json.NewDecoder(r.Body).Decode(&gotRequest); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_ = json.NewEncoder(w).Encode(embedResponse{
			Embeddings: [][]float64{{1, 0}, {0, 1}},
		})
	}))
	defer server.Close()

	client := NewEmbedClient(server.URL, "test-model")
	vectors, err := client.EmbedBatch(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}

	if gotRequest.Model != "test-model" {
		t.Fatalf("model mismatch: got=%s", gotRequest.Model)
	}
	if len(gotRequest.Input) != 2 || gotRequest.Input[0] != "first" || gotRequest.Input[1] != "second" {
		t.Fatalf("input mismatch: %+v", gotRequest.Input)
	}
	if len(vectors) != 2 || vectors[0][0] != 1 || vectors[1][1] != 1 {
		t.Fatalf("unexpected vectors: %+v", vectors)
	}
}

func TestEmbedClientEmbedUsesSingleBatchResult(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(embedResponse{Embeddings: [][]float64{{0.25, 0.75}}})
	}))
	defer server.Close()

	vector, err := NewEmbedClient(server.URL, "model").Embed(context.Background(), "text")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vector) != 2 || vector[0] != 0.25 || vector[1] != 0.75 {
		t.Fatalf("unexpected vector: %+v", vector)
	}
}

func TestEmbedClientErrors(t *testing.T) {
	if _, err := NewEmbedClient("http://127.0.0.1:1", "model").EmbedBatch(context.Background(), nil); err == nil {
		t.Fatalf("expected empty texts error")
	}

	t.Run("non 2xx", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "bad", http.StatusBadGateway)
		}))
		defer server.Close()

		_, err := NewEmbedClient(server.URL, "model").EmbedBatch(context.Background(), []string{"a"})
		if err == nil || !strings.Contains(err.Error(), "embed endpoint returned") {
			t.Fatalf("expected status error, got %v", err)
		}
	})

	t.Run("invalid json", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte("{"))
		}))
		defer server.Close()

		_, err := NewEmbedClient(server.URL, "model").EmbedBatch(context.Background(), []string{"a"})
		if err == nil || !strings.Contains(err.Error(), "decode embed response") {
			t.Fatalf("expected decode error, got %v", err)
		}
	})

	t.Run("embedding count mismatch", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_ = json.NewEncoder(w).Encode(embedResponse{Embeddings: [][]float64{{1, 0}}})
		}))
		defer server.Close()

		_, err := NewEmbedClient(server.URL, "model").EmbedBatch(context.Background(), []string{"a", "b"})
		if err == nil || !strings.Contains(err.Error(), "embedding count mismatch") {
			t.Fatalf("expected mismatch error, got %v", err)
		}
	})

	t.Run("bad URL", func(t *testing.T) {
		_, err := NewEmbedClient("://bad-url", "model").EmbedBatch(context.Background(), []string{"a"})
		if err == nil || !strings.Contains(err.Error(), "new embed request") {
			t.Fatalf("expected bad URL error, got %v", err)
		}
	})
}
