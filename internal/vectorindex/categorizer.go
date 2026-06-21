package vectorindex

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// Embedder embeds a single text into a vector.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float64, error)
}

// BatchEmbedder embeds multiple texts in a single call.
type BatchEmbedder interface {
	EmbedBatch(ctx context.Context, texts []string) ([][]float64, error)
}

type categoryEntry struct {
	name      string
	embedding []float64
}

// Categorizer classifies text against a fixed set of predefined categories.
type Categorizer struct {
	embedder   Embedder
	categories []categoryEntry
}

// CategorizeResult is the output of a categorization request.
type CategorizeResult struct {
	Category   string  `json:"category"`
	Similarity float64 `json:"similarity"`
	Distance   float64 `json:"distance"`
}

// ParseCategoriesFile parses a categories file where each line has the form:
//
//	category-name: description text used for embedding
//
// Lines starting with '#' and blank lines are ignored.
func ParseCategoriesFile(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open categories file: %w", err)
	}
	defer f.Close()
	return parseCategories(f)
}

func parseCategories(r io.Reader) (map[string]string, error) {
	categories := make(map[string]string)
	scanner := bufio.NewScanner(r)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		idx := strings.Index(line, ":")
		if idx <= 0 {
			return nil, fmt.Errorf("line %d: invalid format (expected 'name: description'): %q", lineNum, line)
		}
		name := strings.TrimSpace(line[:idx])
		desc := strings.TrimSpace(line[idx+1:])
		if name == "" {
			return nil, fmt.Errorf("line %d: empty category name", lineNum)
		}
		if desc == "" {
			return nil, fmt.Errorf("line %d: empty description for category %q", lineNum, name)
		}
		categories[name] = desc
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan categories file: %w", err)
	}
	if len(categories) == 0 {
		return nil, fmt.Errorf("no categories found")
	}
	return categories, nil
}

// NewCategorizer embeds all category descriptions and returns a ready-to-use Categorizer.
func NewCategorizer(ctx context.Context, embedder Embedder, categories map[string]string) (*Categorizer, error) {
	if embedder == nil {
		return nil, fmt.Errorf("embedder is nil")
	}
	if len(categories) == 0 {
		return nil, fmt.Errorf("categories are required")
	}

	names := make([]string, 0, len(categories))
	descs := make([]string, 0, len(categories))
	for name, desc := range categories {
		names = append(names, name)
		descs = append(descs, desc)
	}

	embeddings, err := embedAll(ctx, embedder, descs)
	if err != nil {
		return nil, fmt.Errorf("embed categories: %w", err)
	}

	entries := make([]categoryEntry, len(names))
	for i := range names {
		vec := embeddings[i]
		normalizeInPlace(vec)
		entries[i] = categoryEntry{name: names[i], embedding: vec}
	}

	return &Categorizer{embedder: embedder, categories: entries}, nil
}

// Categorize embeds text and returns the closest predefined category with its similarity and distance.
func (c *Categorizer) Categorize(ctx context.Context, text string) (CategorizeResult, error) {
	if text == "" {
		return CategorizeResult{}, fmt.Errorf("text is required")
	}

	vec, err := c.embedder.Embed(ctx, text)
	if err != nil {
		return CategorizeResult{}, fmt.Errorf("embed text: %w", err)
	}
	if len(vec) == 0 {
		return CategorizeResult{}, fmt.Errorf("embedding is empty")
	}
	normalizeInPlace(vec)

	bestSim := math.Inf(-1)
	bestName := ""
	for _, entry := range c.categories {
		sim := dotProduct(vec, entry.embedding)
		if sim > bestSim {
			bestSim = sim
			bestName = entry.name
		}
	}

	return CategorizeResult{
		Category:   bestName,
		Similarity: bestSim,
		Distance:   1 - bestSim,
	}, nil
}

// CategoryNames returns the names of all loaded categories.
func (c *Categorizer) CategoryNames() []string {
	names := make([]string, len(c.categories))
	for i, e := range c.categories {
		names[i] = e.name
	}
	return names
}

func embedAll(ctx context.Context, embedder Embedder, texts []string) ([][]float64, error) {
	if be, ok := embedder.(BatchEmbedder); ok {
		return be.EmbedBatch(ctx, texts)
	}
	vecs := make([][]float64, len(texts))
	for i, text := range texts {
		var err error
		vecs[i], err = embedder.Embed(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("embed at index %d: %w", i, err)
		}
	}
	return vecs, nil
}

func dotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot float64
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

func normalizeInPlace(vec []float64) {
	norm := vectorNorm(vec)
	if norm == 0 {
		return
	}
	for i := range vec {
		vec[i] /= norm
	}
}

func vectorNorm(vec []float64) float64 {
	var sum float64
	for _, v := range vec {
		sum += v * v
	}
	return math.Sqrt(sum)
}
