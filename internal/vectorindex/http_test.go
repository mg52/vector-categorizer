package vectorindex

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCategorizeEndpointJSONBody(t *testing.T) {
	handler := NewHTTP(newTestCategorizer(t))

	req := httptest.NewRequest(http.MethodPost, "/categorize", bytes.NewBufferString(`{"text":"pasta recipe dinner"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.Categorize(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}
	var result CategorizeResult
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
	handler := NewHTTP(newTestCategorizer(t))

	req := httptest.NewRequest(http.MethodPost, "/categorize", strings.NewReader("hotel booking Paris"))
	rec := httptest.NewRecorder()

	handler.Categorize(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}
	var result CategorizeResult
	if err := json.NewDecoder(rec.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.Category != "travel" {
		t.Fatalf("unexpected category: %q", result.Category)
	}
}

func TestCategorizeRejectsBadRequests(t *testing.T) {
	handler := NewHTTP(newTestCategorizer(t))

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
			handler.Categorize(rec, req)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
			}
		})
	}
}

func TestCategoriesEndpoint(t *testing.T) {
	handler := NewHTTP(newTestCategorizer(t))

	req := httptest.NewRequest(http.MethodGet, "/categories", nil)
	rec := httptest.NewRecorder()

	handler.Categories(rec, req)

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
	handler := NewHTTP(newTestCategorizer(t))

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	handler.Health(rec, req)

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
	handler := NewHTTP(newTestCategorizer(t))

	tests := []struct {
		name    string
		method  string
		target  string
		handler http.HandlerFunc
	}{
		{name: "categorize GET", method: http.MethodGet, target: "/categorize", handler: handler.Categorize},
		{name: "categories POST", method: http.MethodPost, target: "/categories", handler: handler.Categories},
		{name: "health POST", method: http.MethodPost, target: "/health", handler: handler.Health},
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
	handler := NewHTTP(newTestCategorizer(t))

	// text with no embedding in fakeEmbedder → Categorize returns error → 500
	req := httptest.NewRequest(http.MethodPost, "/categorize", strings.NewReader("unknown text with no embedding"))
	rec := httptest.NewRecorder()
	handler.Categorize(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d: %s", rec.Code, rec.Body.String())
	}
}
