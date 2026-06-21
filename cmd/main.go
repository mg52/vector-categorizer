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
	embedURL := getenv("VECTOR_INDEX_EMBED_URL", vectorindex.DefaultEmbedURL)
	embedModel := getenv("VECTOR_INDEX_EMBED_MODEL", vectorindex.DefaultEmbedModel)
	categoriesFile := getenv("VECTOR_INDEX_CATEGORIES_FILE", "categories.txt")

	categories, err := vectorindex.ParseCategoriesFile(categoriesFile)
	if err != nil {
		log.Fatalf("load categories from %s: %v", categoriesFile, err)
	}
	log.Printf("Loaded %d categories from %s", len(categories), categoriesFile)

	embedder := vectorindex.NewEmbedClient(embedURL, embedModel)

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
		log.Printf("Starting vector categorizer on %s (embed=%s model=%s)", addr, embedURL, embedModel)
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

func getenv(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
