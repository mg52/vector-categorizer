package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mg52/vector-categorizer/internal/categorizer"
)

// fakeEmbedder maps text → vector, implements both Embedder and BatchEmbedder.
type fakeEmbedder map[string][]float64

func (f fakeEmbedder) Embed(_ context.Context, text string) ([]float64, error) {
	vec, ok := f[text]
	if !ok {
		return nil, fmt.Errorf("no embedding for %q", text)
	}
	return append([]float64(nil), vec...), nil
}

func (f fakeEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	vecs := make([][]float64, len(texts))
	for i, text := range texts {
		vec, err := f.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		vecs[i] = vec
	}
	return vecs, nil
}

const (
	foodDesc   = "food, cooking, recipe"
	travelDesc = "travel, trip, journey"
	sportsDesc = "sports, exercise, fitness"
)

var testCategories = map[string]string{
	"food":   foodDesc,
	"travel": travelDesc,
	"sports": sportsDesc,
}

var testEmbedder = fakeEmbedder{
	foodDesc:   {1, 0, 0},
	travelDesc: {0, 1, 0},
	sportsDesc: {0, 0, 1},
	// query texts
	"pasta recipe dinner": {0.9, 0.1, 0},
	"hotel booking Paris": {0.1, 0.9, 0},
}

func newTestCategorizer(t *testing.T) *categorizer.Categorizer {
	t.Helper()
	c, err := categorizer.NewCategorizer(context.Background(), testEmbedder, testCategories)
	if err != nil {
		t.Fatalf("NewCategorizer: %v", err)
	}
	return c
}

func absFloat(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func TestCategorizeEndpointJSONBody(t *testing.T) {
	h := NewHTTP(newTestCategorizer(t))

	req := httptest.NewRequest(http.MethodPost, "/categorize", bytes.NewBufferString(`{"text":"pasta recipe dinner"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.Categorize(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}
	var result categorizer.CategorizeResult
	if err := json.NewDecoder(rec.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.Category != "food" {
		t.Fatalf("unexpected category: %q", result.Category)
	}
	if result.Similarity <= 0 {
		t.Fatalf("expected positive similarity, got %f", result.Similarity)
	}
	if absFloat(result.Distance-(1-result.Similarity)) > 1e-9 {
		t.Fatalf("distance mismatch: sim=%f dist=%f", result.Similarity, result.Distance)
	}
}

func TestCategorizeEndpointRawBody(t *testing.T) {
	h := NewHTTP(newTestCategorizer(t))

	req := httptest.NewRequest(http.MethodPost, "/categorize", strings.NewReader("hotel booking Paris"))
	rec := httptest.NewRecorder()

	h.Categorize(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}
	var result categorizer.CategorizeResult
	if err := json.NewDecoder(rec.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.Category != "travel" {
		t.Fatalf("unexpected category: %q", result.Category)
	}
}

func TestCategorizeRejectsBadRequests(t *testing.T) {
	h := NewHTTP(newTestCategorizer(t))

	tests := []struct {
		name   string
		body   string
		header string
	}{
		{name: "bad json", body: "{", header: "application/json"},
		{name: "empty json text", body: `{"text":"   "}`, header: "application/json"},
		{name: "empty raw body", body: "   ", header: "text/plain"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/categorize", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", tt.header)
			rec := httptest.NewRecorder()
			h.Categorize(rec, req)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
			}
		})
	}
}

func TestCategoriesEndpoint(t *testing.T) {
	h := NewHTTP(newTestCategorizer(t))

	req := httptest.NewRequest(http.MethodGet, "/categories", nil)
	rec := httptest.NewRecorder()

	h.Categories(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}
	var response struct {
		Categories []string `json:"categories"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(response.Categories) != len(testCategories) {
		t.Fatalf("expected %d categories, got %d: %v", len(testCategories), len(response.Categories), response.Categories)
	}
}

func TestHealthEndpoint(t *testing.T) {
	h := NewHTTP(newTestCategorizer(t))

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	h.Health(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"status":"ok"`) {
		t.Fatalf("unexpected health body: %s", rec.Body.String())
	}
}

func TestRegisterRoutesAndHealth(t *testing.T) {
	mux := http.NewServeMux()
	NewHTTP(newTestCategorizer(t)).Register(mux)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("health status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestHandlersRejectWrongMethods(t *testing.T) {
	h := NewHTTP(newTestCategorizer(t))

	tests := []struct {
		name    string
		method  string
		target  string
		handler http.HandlerFunc
	}{
		{name: "categorize GET", method: http.MethodGet, target: "/categorize", handler: h.Categorize},
		{name: "categories POST", method: http.MethodPost, target: "/categories", handler: h.Categories},
		{name: "health POST", method: http.MethodPost, target: "/health", handler: h.Health},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.target, nil)
			rec := httptest.NewRecorder()
			tt.handler(rec, req)
			if rec.Code != http.StatusMethodNotAllowed {
				t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
			}
		})
	}
}

func TestCategorizeInternalError(t *testing.T) {
	h := NewHTTP(newTestCategorizer(t))

	req := httptest.NewRequest(http.MethodPost, "/categorize", strings.NewReader("unknown text with no embedding"))
	rec := httptest.NewRecorder()
	h.Categorize(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d: %s", rec.Code, rec.Body.String())
	}
}
