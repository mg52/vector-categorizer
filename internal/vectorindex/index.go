package vectorindex

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
)

const (
	DefaultCategoryCount       = 100
	DefaultSimilarityThreshold = 0.59
	DefaultEmbeddingDimension  = 768
)

var ErrBelowThreshold = errors.New("best category similarity is below threshold")

type Embedder interface {
	Embed(ctx context.Context, text string) ([]float64, error)
}

type BatchEmbedder interface {
	EmbedBatch(ctx context.Context, texts []string) ([][]float64, error)
}

type Category struct {
	ID            int       `json:"id"`
	Name          string    `json:"category,omitempty"`
	Centroid      []float64 `json:"centroid,omitempty"`
	DocumentCount int       `json:"documentCount"`
	centroidNorm  float64
}

type IndexDocumentInput struct {
	ID       string `json:"id,omitempty"`
	Text     string `json:"text"`
	Category string `json:"category,omitempty"`
}

type IndexResult struct {
	DocumentID       string    `json:"documentID"`
	CategoryID       int       `json:"categoryID"`
	Category         string    `json:"category,omitempty"`
	Similarity       float64   `json:"similarity"`
	CategoryDocCount int       `json:"categoryDocumentCount"`
	SeededCategory   bool      `json:"seededCategory"`
	Centroid         []float64 `json:"centroid,omitempty"`
}

type SearchResult struct {
	CategoryID       int       `json:"categoryID"`
	Category         string    `json:"category,omitempty"`
	Similarity       float64   `json:"similarity"`
	AboveThreshold   bool      `json:"aboveThreshold"`
	CategoryDocCount int       `json:"categoryDocumentCount"`
	Centroid         []float64 `json:"centroid,omitempty"`
}

type Indexer struct {
	mu                    sync.RWMutex
	embedder              Embedder
	similarityThreshold   float64
	categories            []Category
	categoryByName        map[string]int
	documentCategories    map[string]int
	documentCategoryNames map[string]string
	nextDocumentID        atomic.Uint64
}

func NewIndexer(embedder Embedder, categoryCount int, similarityThreshold float64) (*Indexer, error) {
	if embedder == nil {
		return nil, fmt.Errorf("embedder is nil")
	}
	if categoryCount <= 0 {
		categoryCount = DefaultCategoryCount
	}
	if similarityThreshold <= -1 || similarityThreshold > 1 {
		return nil, fmt.Errorf("similarity threshold must be in (-1, 1], got %f", similarityThreshold)
	}

	categories := make([]Category, categoryCount)
	for i := range categories {
		categories[i] = Category{ID: i}
	}

	return &Indexer{
		embedder:              embedder,
		similarityThreshold:   similarityThreshold,
		categories:            categories,
		categoryByName:        make(map[string]int),
		documentCategories:    make(map[string]int),
		documentCategoryNames: make(map[string]string),
	}, nil
}

func (idx *Indexer) IndexDocument(ctx context.Context, documentID, text string, includeCentroid bool) (IndexResult, error) {
	results, err := idx.IndexDocuments(ctx, []IndexDocumentInput{{ID: documentID, Text: text}}, includeCentroid)
	if err != nil {
		return IndexResult{}, err
	}
	return results[0], nil
}

func (idx *Indexer) IndexDocuments(ctx context.Context, documents []IndexDocumentInput, includeCentroid bool) ([]IndexResult, error) {
	if len(documents) == 0 {
		return nil, fmt.Errorf("documents are required")
	}

	texts := make([]string, len(documents))
	for i, document := range documents {
		if document.Text == "" {
			return nil, fmt.Errorf("document text is empty at index %d", i)
		}
		texts[i] = document.Text
	}

	vectors, err := idx.embedDocuments(ctx, texts)
	if err != nil {
		return nil, err
	}
	if len(vectors) != len(documents) {
		return nil, fmt.Errorf("embedding count mismatch: got %d want %d", len(vectors), len(documents))
	}
	for i := range vectors {
		if len(vectors[i]) == 0 {
			return nil, fmt.Errorf("embedding is empty at index %d", i)
		}
		normalizeInPlace(vectors[i])
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	results := make([]IndexResult, 0, len(documents))
	for i, document := range documents {
		documentID := document.ID
		if documentID == "" {
			documentID = idx.newDocumentID()
		}

		categoryID, similarity, seeded, err := idx.selectCategoryForDocumentLocked(document.Category, vectors[i])
		if err != nil {
			return nil, err
		}

		category := &idx.categories[categoryID]
		updateCentroid(category, vectors[i])
		idx.documentCategories[documentID] = categoryID
		if category.Name != "" {
			idx.documentCategoryNames[documentID] = category.Name
		} else {
			delete(idx.documentCategoryNames, documentID)
		}

		result := IndexResult{
			DocumentID:       documentID,
			CategoryID:       categoryID,
			Category:         category.Name,
			Similarity:       similarity,
			CategoryDocCount: category.DocumentCount,
			SeededCategory:   seeded,
		}
		if includeCentroid {
			result.Centroid = append([]float64(nil), category.Centroid...)
		}
		results = append(results, result)
	}

	return results, nil
}

func (idx *Indexer) Search(ctx context.Context, text string, includeCentroid bool) (SearchResult, error) {
	if text == "" {
		return SearchResult{}, fmt.Errorf("document text is empty")
	}

	vector, err := idx.embedder.Embed(ctx, text)
	if err != nil {
		return SearchResult{}, fmt.Errorf("embed document: %w", err)
	}
	if len(vector) == 0 {
		return SearchResult{}, fmt.Errorf("embedding is empty")
	}
	normalizeInPlace(vector)

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	categoryID, similarity, ok := idx.bestCategoryLocked(vector)
	if !ok {
		return SearchResult{}, fmt.Errorf("no indexed categories")
	}

	category := idx.categories[categoryID]
	result := SearchResult{
		CategoryID:       category.ID,
		Category:         category.Name,
		Similarity:       similarity,
		AboveThreshold:   similarity >= idx.similarityThreshold,
		CategoryDocCount: category.DocumentCount,
	}
	if includeCentroid {
		result.Centroid = append([]float64(nil), category.Centroid...)
	}
	return result, nil
}

func (idx *Indexer) Categories(includeCentroids bool) []Category {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	out := make([]Category, len(idx.categories))
	for i, category := range idx.categories {
		out[i] = Category{
			ID:            category.ID,
			Name:          category.Name,
			DocumentCount: category.DocumentCount,
		}
		if includeCentroids && len(category.Centroid) > 0 {
			out[i].Centroid = append([]float64(nil), category.Centroid...)
		}
	}
	return out
}

func (idx *Indexer) DocumentCategory(documentID string) (int, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	categoryID, ok := idx.documentCategories[documentID]
	return categoryID, ok
}

func (idx *Indexer) DocumentCategoryName(documentID string) (string, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	categoryName, ok := idx.documentCategoryNames[documentID]
	return categoryName, ok
}

func (idx *Indexer) embedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if batchEmbedder, ok := idx.embedder.(BatchEmbedder); ok {
		vectors, err := batchEmbedder.EmbedBatch(ctx, texts)
		if err != nil {
			return nil, fmt.Errorf("embed documents: %w", err)
		}
		return vectors, nil
	}

	vectors := make([][]float64, len(texts))
	for i, text := range texts {
		vector, err := idx.embedder.Embed(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("embed document at index %d: %w", i, err)
		}
		vectors[i] = vector
	}
	return vectors, nil
}

func (idx *Indexer) selectCategoryForDocumentLocked(categoryName string, vector []float64) (categoryID int, similarity float64, seeded bool, err error) {
	if categoryName != "" {
		return idx.selectNamedCategoryLocked(categoryName, vector)
	}
	return idx.selectCategoryLocked(vector)
}

func (idx *Indexer) selectNamedCategoryLocked(categoryName string, vector []float64) (categoryID int, similarity float64, seeded bool, err error) {
	if categoryID, ok := idx.categoryByName[categoryName]; ok {
		category := &idx.categories[categoryID]
		return categoryID, cosineSimilarityNormalized(vector, category.Centroid, category.centroidNorm), false, nil
	}

	for i := range idx.categories {
		category := &idx.categories[i]
		if category.DocumentCount == 0 && category.Name == "" {
			category.Name = categoryName
			idx.categoryByName[categoryName] = category.ID
			return category.ID, 1, true, nil
		}
	}

	return -1, 0, false, fmt.Errorf("category %q requires a new slot but all categories are full", categoryName)
}

func (idx *Indexer) selectCategoryLocked(vector []float64) (categoryID int, similarity float64, seeded bool, err error) {
	bestCategoryID, bestSimilarity, ok := idx.bestCategoryLocked(vector)
	if !ok {
		return 0, 1, true, nil
	}
	if bestSimilarity >= idx.similarityThreshold {
		return bestCategoryID, bestSimilarity, false, nil
	}
	if firstEmptyCategoryID := idx.firstEmptyCategoryIDLocked(); firstEmptyCategoryID != -1 {
		return firstEmptyCategoryID, 1, true, nil
	}

	return -1, bestSimilarity, false, fmt.Errorf("%w: best=%f threshold=%f", ErrBelowThreshold, bestSimilarity, idx.similarityThreshold)
}

func (idx *Indexer) bestCategoryLocked(vector []float64) (categoryID int, similarity float64, ok bool) {
	bestCategoryID := -1
	bestSimilarity := -1.0

	for i := range idx.categories {
		category := &idx.categories[i]
		if category.DocumentCount == 0 {
			continue
		}

		sim := cosineSimilarityNormalized(vector, category.Centroid, category.centroidNorm)
		if sim > bestSimilarity {
			bestSimilarity = sim
			bestCategoryID = category.ID
		}
	}

	if bestCategoryID == -1 {
		return -1, 0, false
	}
	return bestCategoryID, bestSimilarity, true
}

func (idx *Indexer) firstEmptyCategoryIDLocked() int {
	for i := range idx.categories {
		category := &idx.categories[i]
		if category.DocumentCount == 0 && category.Name == "" {
			return category.ID
		}
	}
	return -1
}

func updateCentroid(category *Category, vector []float64) {
	if category.DocumentCount == 0 {
		category.Centroid = append([]float64(nil), vector...)
		category.DocumentCount = 1
		category.centroidNorm = vectorNorm(category.Centroid)
		return
	}

	oldCount := float64(category.DocumentCount)
	newCount := oldCount + 1
	for i := range category.Centroid {
		category.Centroid[i] = (category.Centroid[i]*oldCount + vector[i]) / newCount
	}
	category.DocumentCount++
	category.centroidNorm = vectorNorm(category.Centroid)
}

func cosineSimilarityNormalized(aUnit, b []float64, bNorm float64) float64 {
	if len(aUnit) != len(b) || len(aUnit) == 0 || bNorm == 0 {
		return 0
	}

	var dot float64
	for i := range aUnit {
		dot += aUnit[i] * b[i]
	}
	return dot / bNorm
}

func normalizeInPlace(vector []float64) {
	norm := vectorNorm(vector)
	if norm == 0 {
		return
	}
	for i := range vector {
		vector[i] /= norm
	}
}

func vectorNorm(vector []float64) float64 {
	var sum float64
	for _, value := range vector {
		sum += value * value
	}
	return math.Sqrt(sum)
}

func (idx *Indexer) newDocumentID() string {
	id := idx.nextDocumentID.Add(1)
	return fmt.Sprintf("doc-%d", id)
}
