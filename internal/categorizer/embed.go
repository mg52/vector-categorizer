package categorizer

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

const (
	DefaultEmbedURL   = "http://localhost:11434/api/embed"
	DefaultEmbedModel = "nomic-embed-text"
)

type EmbedClient struct {
	URL    string
	Model  string
	Client *http.Client
}

type embedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

func NewEmbedClient(url, model string) *EmbedClient {
	if url == "" {
		url = DefaultEmbedURL
	}
	if model == "" {
		model = DefaultEmbedModel
	}

	return &EmbedClient{
		URL:   url,
		Model: model,
		Client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

func (e *EmbedClient) Embed(ctx context.Context, text string) ([]float64, error) {
	embeddings, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) != 1 {
		return nil, fmt.Errorf("embedding count mismatch: got %d want 1", len(embeddings))
	}
	return embeddings[0], nil
}

func (e *EmbedClient) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("texts are required")
	}

	body, err := json.Marshal(embedRequest{
		Model: e.Model,
		Input: texts,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.URL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("new embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := e.Client
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("post embed request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("embed endpoint returned %s", resp.Status)
	}

	var out embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}
	if len(out.Embeddings) != len(texts) {
		return nil, fmt.Errorf("embedding count mismatch: got %d want %d", len(out.Embeddings), len(texts))
	}
	return out.Embeddings, nil
}
