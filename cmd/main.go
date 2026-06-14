package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/mg52/vector-categorizer/internal/vectorindex"
)

func main() {
	addr := getenv("VECTOR_INDEX_ADDR", ":8090")
	categoryCount := getenvInt("VECTOR_INDEX_CATEGORY_COUNT", vectorindex.DefaultCategoryCount)
	threshold := getenvFloat("VECTOR_INDEX_SIMILARITY_THRESHOLD", vectorindex.DefaultSimilarityThreshold)
	embedURL := getenv("VECTOR_INDEX_EMBED_URL", vectorindex.DefaultEmbedURL)
	embedModel := getenv("VECTOR_INDEX_EMBED_MODEL", vectorindex.DefaultEmbedModel)

	embedder := vectorindex.NewEmbedClient(embedURL, embedModel)
	indexer, err := vectorindex.NewIndexer(embedder, categoryCount, threshold)
	if err != nil {
		log.Fatalf("create indexer: %v", err)
	}

	mux := http.NewServeMux()
	vectorindex.NewHTTP(indexer).Register(mux)

	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		log.Printf("Starting vector index service on %s", addr)
		log.Printf("Categories=%d threshold=%.4f embed=%s model=%s", categoryCount, threshold, embedURL, embedModel)
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

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)
}

func getenv(name, fallback string) string {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	return value
}

func getenvInt(name string, fallback int) int {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		log.Printf("Invalid %s=%q, using %d", name, value, fallback)
		return fallback
	}
	return parsed
}

func getenvFloat(name string, fallback float64) float64 {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		log.Printf("Invalid %s=%q, using %.4f", name, value, fallback)
		return fallback
	}
	return parsed
}
