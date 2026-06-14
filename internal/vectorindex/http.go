package vectorindex

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type HTTP struct {
	indexer *Indexer
}

type IndexRequest struct {
	ID              string `json:"id"`
	Text            string `json:"text"`
	Category        string `json:"category,omitempty"`
	IncludeCentroid bool   `json:"includeCentroid"`
}

type BatchIndexRequest struct {
	Documents       []IndexDocumentInput `json:"documents"`
	Texts           []string             `json:"texts"`
	IncludeCentroid bool                 `json:"includeCentroid"`
}

type BatchIndexResponse struct {
	Results []IndexResult `json:"results"`
}

type SearchRequest struct {
	Text            string `json:"text"`
	IncludeCentroid bool   `json:"includeCentroid"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func NewHTTP(indexer *Indexer) *HTTP {
	return &HTTP{indexer: indexer}
}

func (h *HTTP) Register(mux *http.ServeMux) {
	mux.HandleFunc("/index", h.Index)
	mux.HandleFunc("/index/batch", h.BatchIndex)
	mux.HandleFunc("/index/search", h.Search)
	mux.HandleFunc("/categories", h.Categories)
	mux.HandleFunc("/document-category", h.DocumentCategory)
	mux.HandleFunc("/health", h.Health)
}

func (h *HTTP) Index(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	req, err := decodeIndexRequest(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	results, err := h.indexer.IndexDocuments(r.Context(), []IndexDocumentInput{{
		ID:       req.ID,
		Text:     req.Text,
		Category: req.Category,
	}}, req.IncludeCentroid)
	if err != nil {
		if errors.Is(err, ErrBelowThreshold) {
			writeError(w, http.StatusUnprocessableEntity, err.Error())
			return
		}
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, results[0])
}

func (h *HTTP) BatchIndex(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	req, err := decodeBatchIndexRequest(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	results, err := h.indexer.IndexDocuments(r.Context(), req.Documents, req.IncludeCentroid)
	if err != nil {
		if errors.Is(err, ErrBelowThreshold) {
			writeError(w, http.StatusUnprocessableEntity, err.Error())
			return
		}
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, BatchIndexResponse{Results: results})
}

func (h *HTTP) Search(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	req, err := decodeSearchRequest(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	result, err := h.indexer.Search(r.Context(), req.Text, req.IncludeCentroid)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, result)
}

func (h *HTTP) Categories(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	includeCentroids := r.URL.Query().Get("centroids") == "true"
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"categories": h.indexer.Categories(includeCentroids),
	})
}

func (h *HTTP) DocumentCategory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	documentID := r.URL.Query().Get("id")
	if documentID == "" {
		writeError(w, http.StatusBadRequest, "missing id query parameter")
		return
	}

	categoryID, ok := h.indexer.DocumentCategory(documentID)
	if !ok {
		writeError(w, http.StatusNotFound, "document not found")
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"documentID": documentID,
		"categoryID": categoryID,
		"category":   categoryNameOrEmpty(h.indexer, documentID),
	})
}

func (h *HTTP) Health(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func decodeIndexRequest(r *http.Request) (IndexRequest, error) {
	defer r.Body.Close()

	contentType := r.Header.Get("Content-Type")
	if strings.Contains(contentType, "application/json") {
		var req IndexRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			return IndexRequest{}, fmt.Errorf("decode json body: %w", err)
		}
		req.Text = strings.TrimSpace(req.Text)
		req.Category = strings.TrimSpace(req.Category)
		if req.Text == "" {
			return IndexRequest{}, fmt.Errorf("text is required")
		}
		return req, nil
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 8*1024*1024))
	if err != nil {
		return IndexRequest{}, fmt.Errorf("read body: %w", err)
	}
	text := strings.TrimSpace(string(body))
	if text == "" {
		return IndexRequest{}, fmt.Errorf("request body is empty")
	}

	return IndexRequest{
		ID:              r.URL.Query().Get("id"),
		Text:            text,
		Category:        strings.TrimSpace(r.URL.Query().Get("category")),
		IncludeCentroid: r.URL.Query().Get("centroid") == "true",
	}, nil
}

func decodeBatchIndexRequest(r *http.Request) (BatchIndexRequest, error) {
	defer r.Body.Close()

	contentType := r.Header.Get("Content-Type")
	if !strings.Contains(contentType, "application/json") {
		return BatchIndexRequest{}, fmt.Errorf("content type must be application/json")
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 16*1024*1024))
	if err != nil {
		return BatchIndexRequest{}, fmt.Errorf("read body: %w", err)
	}
	bodyText := strings.TrimSpace(string(body))
	if bodyText == "" {
		return BatchIndexRequest{}, fmt.Errorf("request body is empty")
	}

	var req BatchIndexRequest
	if strings.HasPrefix(bodyText, "[") {
		documents, err := decodeBatchArray([]byte(bodyText))
		if err != nil {
			return BatchIndexRequest{}, err
		}
		req.Documents = documents
	} else {
		if err := json.Unmarshal([]byte(bodyText), &req); err != nil {
			return BatchIndexRequest{}, fmt.Errorf("decode json body: %w", err)
		}
		if len(req.Documents) == 0 && len(req.Texts) > 0 {
			req.Documents = documentsFromTexts(req.Texts)
		}
	}

	if err := normalizeBatchDocuments(req.Documents); err != nil {
		return BatchIndexRequest{}, err
	}
	return req, nil
}

func decodeSearchRequest(r *http.Request) (SearchRequest, error) {
	defer r.Body.Close()

	contentType := r.Header.Get("Content-Type")
	if strings.Contains(contentType, "application/json") {
		var req SearchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			return SearchRequest{}, fmt.Errorf("decode json body: %w", err)
		}
		req.Text = strings.TrimSpace(req.Text)
		if req.Text == "" {
			return SearchRequest{}, fmt.Errorf("text is required")
		}
		return req, nil
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 8*1024*1024))
	if err != nil {
		return SearchRequest{}, fmt.Errorf("read body: %w", err)
	}
	text := strings.TrimSpace(string(body))
	if text == "" {
		return SearchRequest{}, fmt.Errorf("request body is empty")
	}

	return SearchRequest{
		Text:            text,
		IncludeCentroid: r.URL.Query().Get("centroid") == "true",
	}, nil
}

func decodeBatchArray(body []byte) ([]IndexDocumentInput, error) {
	var texts []string
	if err := json.Unmarshal(body, &texts); err == nil {
		return documentsFromTexts(texts), nil
	}

	var documents []IndexDocumentInput
	if err := json.Unmarshal(body, &documents); err != nil {
		return nil, fmt.Errorf("decode json body: %w", err)
	}
	return documents, nil
}

func documentsFromTexts(texts []string) []IndexDocumentInput {
	documents := make([]IndexDocumentInput, len(texts))
	for i, text := range texts {
		documents[i] = IndexDocumentInput{Text: text}
	}
	return documents
}

func normalizeBatchDocuments(documents []IndexDocumentInput) error {
	if len(documents) == 0 {
		return fmt.Errorf("documents are required")
	}
	for i := range documents {
		documents[i].ID = strings.TrimSpace(documents[i].ID)
		documents[i].Text = strings.TrimSpace(documents[i].Text)
		documents[i].Category = strings.TrimSpace(documents[i].Category)
		if documents[i].Text == "" {
			return fmt.Errorf("document text is required at index %d", i)
		}
	}
	return nil
}

func categoryNameOrEmpty(indexer *Indexer, documentID string) string {
	categoryName, _ := indexer.DocumentCategoryName(documentID)
	return categoryName
}

func writeJSON(w http.ResponseWriter, status int, value interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, errorResponse{Error: message})
}
