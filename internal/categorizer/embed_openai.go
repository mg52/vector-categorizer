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
	DefaultOpenAIEmbedURL   = "https://api.openai.com/v1/embeddings"
	DefaultOpenAIEmbedModel = "text-embedding-3-small"
)

type OpenAIEmbedClient struct {
	URL    string
	Model  string
	APIKey string
	Client *http.Client
}

type openAIEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type openAIEmbedResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

func NewOpenAIEmbedClient(apiKey, model, url string) *OpenAIEmbedClient {
	if model == "" {
		model = DefaultOpenAIEmbedModel
	}
	if url == "" {
		url = DefaultOpenAIEmbedURL
	}
	return &OpenAIEmbedClient{
		URL:    url,
		Model:  model,
		APIKey: apiKey,
		Client: &http.Client{Timeout: 60 * time.Second},
	}
}

func (e *OpenAIEmbedClient) Embed(ctx context.Context, text string) ([]float64, error) {
	embeddings, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) != 1 {
		return nil, fmt.Errorf("embedding count mismatch: got %d want 1", len(embeddings))
	}
	return embeddings[0], nil
}

func (e *OpenAIEmbedClient) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("texts are required")
	}

	body, err := json.Marshal(openAIEmbedRequest{Model: e.Model, Input: texts})
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.URL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("new embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.APIKey)

	resp, err := e.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("post embed request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("openai embed endpoint returned %s", resp.Status)
	}

	var out openAIEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("embedding count mismatch: got %d want %d", len(out.Data), len(texts))
	}

	// OpenAI returns embeddings sorted by index, not necessarily input order
	embeddings := make([][]float64, len(texts))
	for _, d := range out.Data {
		embeddings[d.Index] = d.Embedding
	}
	return embeddings, nil
}
