package vectorindex

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type HTTP struct {
	categorizer *Categorizer
}

type CategorizeRequest struct {
	Text string `json:"text"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func NewHTTP(categorizer *Categorizer) *HTTP {
	return &HTTP{categorizer: categorizer}
}

func (h *HTTP) Register(mux *http.ServeMux) {
	mux.HandleFunc("/categorize", h.Categorize)
	mux.HandleFunc("/categories", h.Categories)
	mux.HandleFunc("/health", h.Health)
}

func (h *HTTP) Categorize(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	text, err := decodeText(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	result, err := h.categorizer.Categorize(r.Context(), text)
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
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"categories": h.categorizer.CategoryNames(),
	})
}

func (h *HTTP) Health(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func decodeText(r *http.Request) (string, error) {
	defer r.Body.Close()

	if strings.Contains(r.Header.Get("Content-Type"), "application/json") {
		var req CategorizeRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			return "", fmt.Errorf("decode json body: %w", err)
		}
		text := strings.TrimSpace(req.Text)
		if text == "" {
			return "", fmt.Errorf("text is required")
		}
		return text, nil
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 8*1024*1024))
	if err != nil {
		return "", fmt.Errorf("read body: %w", err)
	}
	text := strings.TrimSpace(string(body))
	if text == "" {
		return "", fmt.Errorf("request body is empty")
	}
	return text, nil
}

func writeJSON(w http.ResponseWriter, status int, value interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, errorResponse{Error: message})
}
