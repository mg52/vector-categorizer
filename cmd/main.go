package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/mg52/vector-categorizer/internal/vectorindex"
)

func main() {
	addr := getenv("VECTOR_INDEX_ADDR", ":8090")
	categoriesFile := getenv("VECTOR_INDEX_CATEGORIES_FILE", "categories.txt")
	provider := getenv("VECTOR_INDEX_EMBED_PROVIDER", "ollama") // "ollama" or "openai"

	categories, err := vectorindex.ParseCategoriesFile(categoriesFile)
	if err != nil {
		log.Fatalf("load categories from %s: %v", categoriesFile, err)
	}
	log.Printf("Loaded %d categories from %s", len(categories), categoriesFile)

	embedder := buildEmbedder(provider)

	ctx := context.Background()
	categorizer, err := vectorindex.NewCategorizer(ctx, embedder, categories)
	if err != nil {
		log.Fatalf("create categorizer: %v", err)
	}
	log.Printf("Categories ready: %v", categorizer.CategoryNames())

	mux := http.NewServeMux()
	vectorindex.NewHTTP(categorizer).Register(mux)

	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		log.Printf("Starting vector categorizer on %s (provider=%s)", addr, provider)
		errCh <- srv.ListenAndServe()
	}()

	stopCh := make(chan os.Signal, 1)
	signal.Notify(stopCh, os.Interrupt, syscall.SIGTERM)

	select {
	case sig := <-stopCh:
		log.Printf("Received %s, shutting down", sig)
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("Server failed: %v", err)
		}
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
}

func buildEmbedder(provider string) vectorindex.Embedder {
	switch provider {
	case "openai":
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			log.Fatalf("OPENAI_API_KEY is required when VECTOR_INDEX_EMBED_PROVIDER=openai")
		}
		model := getenv("VECTOR_INDEX_EMBED_MODEL", vectorindex.DefaultOpenAIEmbedModel)
		url := getenv("VECTOR_INDEX_EMBED_URL", vectorindex.DefaultOpenAIEmbedURL)
		client := vectorindex.NewOpenAIEmbedClient(apiKey, model, url)
		log.Printf("Using OpenAI embedder (url=%s model=%s)", client.URL, model)
		return client
	default: // "ollama"
		url := getenv("VECTOR_INDEX_EMBED_URL", vectorindex.DefaultEmbedURL)
		model := getenv("VECTOR_INDEX_EMBED_MODEL", vectorindex.DefaultEmbedModel)
		log.Printf("Using Ollama embedder (url=%s model=%s)", url, model)
		return vectorindex.NewEmbedClient(url, model)
	}
}

func getenv(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
