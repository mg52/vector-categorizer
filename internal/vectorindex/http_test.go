package vectorindex

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestBatchIndexAcceptsTextArray(t *testing.T) {
	indexer := newTestIndexer(t, 3, 0.90)
	handler := NewHTTP(indexer)

	body := bytes.NewBufferString(`["a","a2","b"]`)
	req := httptest.NewRequest(http.MethodPost, "/index/batch", body)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.BatchIndex(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}

	var response BatchIndexResponse
	if err := json.NewDecoder(rec.Body).Decode(&response); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if len(response.Results) != 3 {
		t.Fatalf("result count mismatch: got=%d", len(response.Results))
	}
	if response.Results[0].CategoryID != 0 || response.Results[1].CategoryID != 0 || response.Results[2].CategoryID != 1 {
		t.Fatalf("unexpected categories: %+v", response.Results)
	}
}

func TestBatchIndexAcceptsDocumentObjects(t *testing.T) {
	indexer := newTestIndexer(t, 2, 0.90)
	handler := NewHTTP(indexer)

	body := bytes.NewBufferString(`{"documents":[{"id":"doc-a","text":"a"},{"id":"doc-a2","text":"a2"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/index/batch", body)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.BatchIndex(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}

	var response BatchIndexResponse
	if err := json.NewDecoder(rec.Body).Decode(&response); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if len(response.Results) != 2 {
		t.Fatalf("result count mismatch: got=%d", len(response.Results))
	}
	if response.Results[0].DocumentID != "doc-a" || response.Results[1].DocumentID != "doc-a2" {
		t.Fatalf("unexpected document IDs: %+v", response.Results)
	}
}

func TestBatchIndexWithCategoriesAndSearchEndpoint(t *testing.T) {
	indexer, err := NewIndexer(fakeEmbedder{
		"jacket-a":    {1, 0},
		"jacket-b":    {0.8, 0.2},
		"headphones":  {0, 1},
		"earbuds":     {0.1, 0.9},
		"query-audio": {0.05, 0.95},
	}, 10, 0.60)
	if err != nil {
		t.Fatalf("NewIndexer: %v", err)
	}
	handler := NewHTTP(indexer)

	batchBody := bytes.NewBufferString(`{"documents":[` +
		`{"id":"doc-jacket-a","text":"jacket-a","category":"outdoor"},` +
		`{"id":"doc-jacket-b","text":"jacket-b","category":"outdoor"},` +
		`{"id":"doc-headphones","text":"headphones","category":"audio"},` +
		`{"id":"doc-earbuds","text":"earbuds","category":"audio"}` +
		`]}`)
	batchReq := httptest.NewRequest(http.MethodPost, "/index/batch", batchBody)
	batchReq.Header.Set("Content-Type", "application/json")
	batchRec := httptest.NewRecorder()

	handler.BatchIndex(batchRec, batchReq)

	if batchRec.Code != http.StatusOK {
		t.Fatalf("batch status mismatch: got=%d body=%s", batchRec.Code, batchRec.Body.String())
	}

	var batchResponse BatchIndexResponse
	if err := json.NewDecoder(batchRec.Body).Decode(&batchResponse); err != nil {
		t.Fatalf("decode batch response: %v", err)
	}
	if batchResponse.Results[0].Category != "outdoor" || batchResponse.Results[2].Category != "audio" {
		t.Fatalf("unexpected batch categories: %+v", batchResponse.Results)
	}

	searchBody := bytes.NewBufferString(`{"text":"query-audio"}`)
	searchReq := httptest.NewRequest(http.MethodPost, "/index/search", searchBody)
	searchReq.Header.Set("Content-Type", "application/json")
	searchRec := httptest.NewRecorder()

	handler.Search(searchRec, searchReq)

	if searchRec.Code != http.StatusOK {
		t.Fatalf("search status mismatch: got=%d body=%s", searchRec.Code, searchRec.Body.String())
	}

	var searchResponse SearchResult
	if err := json.NewDecoder(searchRec.Body).Decode(&searchResponse); err != nil {
		t.Fatalf("decode search response: %v", err)
	}
	if searchResponse.Category != "audio" {
		t.Fatalf("expected audio prediction, got %+v", searchResponse)
	}
}

func TestRegisterRoutesAndHealth(t *testing.T) {
	indexer := newTestIndexer(t, 2, 0.90)
	mux := http.NewServeMux()
	NewHTTP(indexer).Register(mux)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("health status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"status":"ok"`) {
		t.Fatalf("unexpected health body: %s", rec.Body.String())
	}
}

func TestHandlersRejectWrongMethods(t *testing.T) {
	indexer := newTestIndexer(t, 2, 0.90)
	handler := NewHTTP(indexer)

	tests := []struct {
		name    string
		method  string
		target  string
		handler http.HandlerFunc
	}{
		{name: "index", method: http.MethodGet, target: "/index", handler: handler.Index},
		{name: "batch", method: http.MethodGet, target: "/index/batch", handler: handler.BatchIndex},
		{name: "search", method: http.MethodGet, target: "/index/search", handler: handler.Search},
		{name: "categories", method: http.MethodPost, target: "/categories", handler: handler.Categories},
		{name: "document category", method: http.MethodPost, target: "/document-category", handler: handler.DocumentCategory},
		{name: "health", method: http.MethodPost, target: "/health", handler: handler.Health},
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

func TestIndexAcceptsRawBodyAndCategoryQuery(t *testing.T) {
	indexer := newTestIndexer(t, 2, 0.90)
	handler := NewHTTP(indexer)

	req := httptest.NewRequest(http.MethodPost, "/index?id=raw-1&category=audio&centroid=true", strings.NewReader(" a "))
	rec := httptest.NewRecorder()

	handler.Index(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}

	var result IndexResult
	if err := json.NewDecoder(rec.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.DocumentID != "raw-1" || result.Category != "audio" || len(result.Centroid) == 0 {
		t.Fatalf("unexpected index result: %+v", result)
	}
}

func TestIndexRejectsBadRequests(t *testing.T) {
	indexer := newTestIndexer(t, 1, 0.90)
	handler := NewHTTP(indexer)

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
			req := httptest.NewRequest(http.MethodPost, "/index", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", tt.header)
			rec := httptest.NewRecorder()
			handler.Index(rec, req)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
			}
		})
	}
}

func TestBatchIndexAcceptsTextsObjectAndRejectsBadRequests(t *testing.T) {
	indexer := newTestIndexer(t, 3, 0.90)
	handler := NewHTTP(indexer)

	req := httptest.NewRequest(http.MethodPost, "/index/batch", strings.NewReader(`{"texts":["a","a2"]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.BatchIndex(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("texts object status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}

	badRequests := []struct {
		name        string
		body        string
		contentType string
	}{
		{name: "wrong content type", body: `{"texts":["a"]}`, contentType: "text/plain"},
		{name: "empty body", body: ` `, contentType: "application/json"},
		{name: "bad json", body: `{`, contentType: "application/json"},
		{name: "empty documents", body: `{"documents":[]}`, contentType: "application/json"},
		{name: "empty text", body: `{"documents":[{"text":" "}]}`, contentType: "application/json"},
	}

	for _, tt := range badRequests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/index/batch", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", tt.contentType)
			rec := httptest.NewRecorder()
			handler.BatchIndex(rec, req)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
			}
		})
	}
}

func TestBatchIndexReturnsUnprocessableWhenBelowThreshold(t *testing.T) {
	indexer := newTestIndexer(t, 1, 0.90)
	handler := NewHTTP(indexer)

	req := httptest.NewRequest(http.MethodPost, "/index/batch", strings.NewReader(`{"texts":["a","c"]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.BatchIndex(rec, req)

	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestSearchAcceptsRawBodyAndRejectsBadRequests(t *testing.T) {
	indexer := newTestIndexer(t, 2, 0.90)
	handler := NewHTTP(indexer)

	if _, err := indexer.IndexDocument(nil, "doc-a", "a", false); err != nil {
		t.Fatalf("IndexDocument: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/index/search?centroid=true", strings.NewReader("a2"))
	rec := httptest.NewRecorder()
	handler.Search(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("search raw status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}
	var result SearchResult
	if err := json.NewDecoder(rec.Body).Decode(&result); err != nil {
		t.Fatalf("decode search response: %v", err)
	}
	if len(result.Centroid) == 0 {
		t.Fatalf("expected centroid in search response")
	}

	badRequests := []struct {
		name   string
		body   string
		header string
	}{
		{name: "bad json", body: `{`, header: "application/json"},
		{name: "empty json text", body: `{"text":" "}`, header: "application/json"},
		{name: "empty raw body", body: " ", header: "text/plain"},
	}
	for _, tt := range badRequests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/index/search", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", tt.header)
			rec := httptest.NewRecorder()
			handler.Search(rec, req)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
			}
		})
	}
}

func TestSearchReturnsUnprocessableWhenNoCategoriesIndexed(t *testing.T) {
	indexer := newTestIndexer(t, 2, 0.90)
	handler := NewHTTP(indexer)

	req := httptest.NewRequest(http.MethodPost, "/index/search", strings.NewReader(`{"text":"a"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.Search(rec, req)

	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status mismatch: got=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestCategoriesAndDocumentCategoryEndpoints(t *testing.T) {
	indexer := newTestIndexer(t, 2, 0.90)
	handler := NewHTTP(indexer)

	if _, err := indexer.IndexDocuments(nil, []IndexDocumentInput{{ID: "doc-a", Text: "a", Category: "audio"}}, false); err != nil {
		t.Fatalf("IndexDocuments: %v", err)
	}

	categoriesReq := httptest.NewRequest(http.MethodGet, "/categories?centroids=true", nil)
	categoriesRec := httptest.NewRecorder()
	handler.Categories(categoriesRec, categoriesReq)
	if categoriesRec.Code != http.StatusOK {
		t.Fatalf("categories status mismatch: got=%d body=%s", categoriesRec.Code, categoriesRec.Body.String())
	}
	if !strings.Contains(categoriesRec.Body.String(), `"category":"audio"`) || !strings.Contains(categoriesRec.Body.String(), `"centroid"`) {
		t.Fatalf("unexpected categories body: %s", categoriesRec.Body.String())
	}

	docReq := httptest.NewRequest(http.MethodGet, "/document-category?id=doc-a", nil)
	docRec := httptest.NewRecorder()
	handler.DocumentCategory(docRec, docReq)
	if docRec.Code != http.StatusOK {
		t.Fatalf("document category status mismatch: got=%d body=%s", docRec.Code, docRec.Body.String())
	}
	if !strings.Contains(docRec.Body.String(), `"category":"audio"`) {
		t.Fatalf("unexpected document category body: %s", docRec.Body.String())
	}

	missingIDReq := httptest.NewRequest(http.MethodGet, "/document-category", nil)
	missingIDRec := httptest.NewRecorder()
	handler.DocumentCategory(missingIDRec, missingIDReq)
	if missingIDRec.Code != http.StatusBadRequest {
		t.Fatalf("missing id status mismatch: got=%d body=%s", missingIDRec.Code, missingIDRec.Body.String())
	}

	notFoundReq := httptest.NewRequest(http.MethodGet, "/document-category?id=missing", nil)
	notFoundRec := httptest.NewRecorder()
	handler.DocumentCategory(notFoundRec, notFoundReq)
	if notFoundRec.Code != http.StatusNotFound {
		t.Fatalf("not found status mismatch: got=%d body=%s", notFoundRec.Code, notFoundRec.Body.String())
	}
}
