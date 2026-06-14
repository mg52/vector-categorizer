package vectorindex

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
)

type fakeEmbedder map[string][]float64

func (f fakeEmbedder) Embed(_ context.Context, text string) ([]float64, error) {
	vector, ok := f[text]
	if !ok {
		return nil, errors.New("missing fake vector")
	}
	return append([]float64(nil), vector...), nil
}

type fakeBatchEmbedder struct {
	fakeEmbedder
	calls   int
	texts   []string
	vectors [][]float64
	err     error
}

func (f *fakeBatchEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	f.calls++
	f.texts = append([]string(nil), texts...)
	if f.err != nil {
		return nil, f.err
	}
	if f.vectors != nil {
		return f.vectors, nil
	}

	vectors := make([][]float64, len(texts))
	for i, text := range texts {
		vector, err := f.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		vectors[i] = vector
	}
	return vectors, nil
}

type errEmbedder struct {
	err error
}

func (e errEmbedder) Embed(_ context.Context, _ string) ([]float64, error) {
	return nil, e.err
}

func TestNewIndexerValidationAndDefaults(t *testing.T) {
	if _, err := NewIndexer(nil, 1, 0.5); err == nil {
		t.Fatalf("expected nil embedder error")
	}
	if _, err := NewIndexer(fakeEmbedder{"a": {1, 0}}, 1, 1.1); err == nil {
		t.Fatalf("expected invalid threshold error")
	}

	indexer, err := NewIndexer(fakeEmbedder{"a": {1, 0}}, 0, 0.5)
	if err != nil {
		t.Fatalf("NewIndexer default category count: %v", err)
	}
	if len(indexer.Categories(false)) != DefaultCategoryCount {
		t.Fatalf("default category count mismatch: got=%d", len(indexer.Categories(false)))
	}
}

func TestIndexDocumentSeedsEmptyCategoriesAndTracksDocuments(t *testing.T) {
	indexer := newTestIndexer(t, 2, 0.90)

	first, err := indexer.IndexDocument(context.Background(), "doc-a", "a", false)
	if err != nil {
		t.Fatalf("IndexDocument first: %v", err)
	}
	if first.CategoryID != 0 || !first.SeededCategory || first.CategoryDocCount != 1 {
		t.Fatalf("unexpected first result: %+v", first)
	}

	second, err := indexer.IndexDocument(context.Background(), "doc-b", "b", false)
	if err != nil {
		t.Fatalf("IndexDocument second: %v", err)
	}
	if second.CategoryID != 1 || !second.SeededCategory || second.CategoryDocCount != 1 {
		t.Fatalf("unexpected second result: %+v", second)
	}

	categoryID, ok := indexer.DocumentCategory("doc-b")
	if !ok || categoryID != 1 {
		t.Fatalf("document category mismatch: got=%d ok=%v", categoryID, ok)
	}
}

func TestIndexDocumentAssignsNearestCategoryAboveThreshold(t *testing.T) {
	indexer := newTestIndexer(t, 2, 0.80)

	if _, err := indexer.IndexDocument(context.Background(), "doc-a", "a", false); err != nil {
		t.Fatalf("IndexDocument first: %v", err)
	}

	result, err := indexer.IndexDocument(context.Background(), "doc-a2", "a2", false)
	if err != nil {
		t.Fatalf("IndexDocument second: %v", err)
	}
	if result.CategoryID != 0 || result.SeededCategory || result.CategoryDocCount != 2 {
		t.Fatalf("unexpected assignment: %+v", result)
	}
	if result.Similarity < 0.99 {
		t.Fatalf("expected high similarity, got %f", result.Similarity)
	}
}

func TestIndexDocumentReturnsBelowThresholdWhenCategoriesAreFull(t *testing.T) {
	indexer := newTestIndexer(t, 1, 0.50)

	if _, err := indexer.IndexDocument(context.Background(), "doc-a", "a", false); err != nil {
		t.Fatalf("IndexDocument first: %v", err)
	}

	_, err := indexer.IndexDocument(context.Background(), "doc-c", "c", false)
	if !errors.Is(err, ErrBelowThreshold) {
		t.Fatalf("expected ErrBelowThreshold, got %v", err)
	}
}

func TestUpdateCentroidUsesIncrementalAverage(t *testing.T) {
	indexer := newTestIndexer(t, 1, 0)

	if _, err := indexer.IndexDocument(context.Background(), "doc-a", "a", false); err != nil {
		t.Fatalf("IndexDocument first: %v", err)
	}
	result, err := indexer.IndexDocument(context.Background(), "doc-c", "c", true)
	if err != nil {
		t.Fatalf("IndexDocument second: %v", err)
	}

	if result.CategoryID != 0 || result.CategoryDocCount != 2 {
		t.Fatalf("unexpected result: %+v", result)
	}
	assertFloatNear(t, result.Centroid[0], 0.5)
	assertFloatNear(t, result.Centroid[1], 0.5)
}

func TestIndexDocumentAutoGeneratesIDAndCentroidCopy(t *testing.T) {
	indexer := newTestIndexer(t, 1, 0)

	result, err := indexer.IndexDocument(context.Background(), "", "a", true)
	if err != nil {
		t.Fatalf("IndexDocument: %v", err)
	}
	if result.DocumentID != "doc-1" {
		t.Fatalf("unexpected generated document ID: %s", result.DocumentID)
	}
	if len(result.Centroid) != 2 {
		t.Fatalf("expected centroid in result: %+v", result)
	}

	result.Centroid[0] = 999
	categories := indexer.Categories(true)
	if categories[0].Centroid[0] == 999 {
		t.Fatalf("result centroid should be a copy")
	}

	categories[0].Centroid[0] = 123
	categoriesAgain := indexer.Categories(true)
	if categoriesAgain[0].Centroid[0] == 123 {
		t.Fatalf("categories centroid should be a copy")
	}
}

func TestIndexDocumentsUsesBatchEmbedder(t *testing.T) {
	embedder := &fakeBatchEmbedder{
		fakeEmbedder: fakeEmbedder{
			"a":  {1, 0},
			"a2": {1, 0},
		},
	}
	indexer, err := NewIndexer(embedder, 2, 0.90)
	if err != nil {
		t.Fatalf("NewIndexer: %v", err)
	}

	results, err := indexer.IndexDocuments(context.Background(), []IndexDocumentInput{
		{ID: "doc-a", Text: "a"},
		{ID: "doc-a2", Text: "a2"},
	}, false)
	if err != nil {
		t.Fatalf("IndexDocuments: %v", err)
	}

	if embedder.calls != 1 {
		t.Fatalf("expected one batch embed call, got %d", embedder.calls)
	}
	if len(embedder.texts) != 2 || embedder.texts[0] != "a" || embedder.texts[1] != "a2" {
		t.Fatalf("unexpected batch texts: %#v", embedder.texts)
	}
	if len(results) != 2 || results[0].CategoryID != 0 || results[1].CategoryID != 0 {
		t.Fatalf("unexpected results: %+v", results)
	}
}

func TestIndexDocumentsValidationAndEmbeddingErrors(t *testing.T) {
	indexer := newTestIndexer(t, 2, 0.90)
	if _, err := indexer.IndexDocuments(context.Background(), nil, false); err == nil {
		t.Fatalf("expected empty documents error")
	}
	if _, err := indexer.IndexDocuments(context.Background(), []IndexDocumentInput{{Text: ""}}, false); err == nil {
		t.Fatalf("expected empty document text error")
	}
	if _, err := indexer.IndexDocuments(context.Background(), []IndexDocumentInput{{Text: "missing"}}, false); err == nil {
		t.Fatalf("expected missing fake vector error")
	}

	mismatchEmbedder := &fakeBatchEmbedder{vectors: [][]float64{{1, 0}}}
	mismatchIndexer, err := NewIndexer(mismatchEmbedder, 2, 0.5)
	if err != nil {
		t.Fatalf("NewIndexer mismatch: %v", err)
	}
	_, err = mismatchIndexer.IndexDocuments(context.Background(), []IndexDocumentInput{{Text: "a"}, {Text: "b"}}, false)
	if err == nil || !strings.Contains(err.Error(), "embedding count mismatch") {
		t.Fatalf("expected embedding count mismatch, got %v", err)
	}

	emptyVectorEmbedder := &fakeBatchEmbedder{vectors: [][]float64{{}}}
	emptyVectorIndexer, err := NewIndexer(emptyVectorEmbedder, 1, 0.5)
	if err != nil {
		t.Fatalf("NewIndexer empty vector: %v", err)
	}
	_, err = emptyVectorIndexer.IndexDocuments(context.Background(), []IndexDocumentInput{{Text: "a"}}, false)
	if err == nil || !strings.Contains(err.Error(), "embedding is empty") {
		t.Fatalf("expected empty embedding error, got %v", err)
	}

	batchErrEmbedder := &fakeBatchEmbedder{err: errors.New("boom")}
	batchErrIndexer, err := NewIndexer(batchErrEmbedder, 1, 0.5)
	if err != nil {
		t.Fatalf("NewIndexer batch error: %v", err)
	}
	_, err = batchErrIndexer.IndexDocuments(context.Background(), []IndexDocumentInput{{Text: "a"}}, false)
	if err == nil || !strings.Contains(err.Error(), "embed documents") {
		t.Fatalf("expected wrapped batch embed error, got %v", err)
	}
}

func TestIndexDocumentsUsesProvidedCategoriesAndSearchesCentroids(t *testing.T) {
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

	results, err := indexer.IndexDocuments(context.Background(), []IndexDocumentInput{
		{ID: "doc-jacket-a", Text: "jacket-a", Category: "outdoor"},
		{ID: "doc-jacket-b", Text: "jacket-b", Category: "outdoor"},
		{ID: "doc-headphones", Text: "headphones", Category: "audio"},
		{ID: "doc-earbuds", Text: "earbuds", Category: "audio"},
	}, false)
	if err != nil {
		t.Fatalf("IndexDocuments: %v", err)
	}

	if results[0].Category != "outdoor" || results[1].Category != "outdoor" {
		t.Fatalf("expected outdoor grouping, got %+v", results[:2])
	}
	if results[2].Category != "audio" || results[3].Category != "audio" {
		t.Fatalf("expected audio grouping, got %+v", results[2:])
	}
	if results[0].CategoryID != results[1].CategoryID {
		t.Fatalf("same category should reuse category ID: %+v", results[:2])
	}
	if results[2].CategoryID != results[3].CategoryID {
		t.Fatalf("same category should reuse category ID: %+v", results[2:])
	}
	if results[1].CategoryDocCount != 2 || results[3].CategoryDocCount != 2 {
		t.Fatalf("unexpected category document counts: %+v", results)
	}

	categoryName, ok := indexer.DocumentCategoryName("doc-earbuds")
	if !ok || categoryName != "audio" {
		t.Fatalf("document category name mismatch: got=%q ok=%v", categoryName, ok)
	}

	prediction, err := indexer.Search(context.Background(), "query-audio", false)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if prediction.Category != "audio" {
		t.Fatalf("expected audio prediction, got %+v", prediction)
	}
	if prediction.CategoryDocCount != 2 {
		t.Fatalf("unexpected predicted category count: %+v", prediction)
	}
}

func TestNamedCategoryReturnsErrorWhenSlotsAreFull(t *testing.T) {
	indexer, err := NewIndexer(fakeEmbedder{
		"a": {1, 0},
		"b": {0, 1},
	}, 1, 0.5)
	if err != nil {
		t.Fatalf("NewIndexer: %v", err)
	}

	if _, err := indexer.IndexDocuments(context.Background(), []IndexDocumentInput{{Text: "a", Category: "audio"}}, false); err != nil {
		t.Fatalf("IndexDocuments first category: %v", err)
	}
	_, err = indexer.IndexDocuments(context.Background(), []IndexDocumentInput{{Text: "b", Category: "outdoor"}}, false)
	if err == nil || !strings.Contains(err.Error(), "all categories are full") {
		t.Fatalf("expected full category slot error, got %v", err)
	}
}

func TestSearchValidationAndThreshold(t *testing.T) {
	indexer := newTestIndexer(t, 1, 0.95)
	if _, err := indexer.Search(context.Background(), "", false); err == nil {
		t.Fatalf("expected empty search text error")
	}
	if _, err := indexer.Search(context.Background(), "a", false); err == nil {
		t.Fatalf("expected no indexed categories error")
	}
	if _, err := indexer.IndexDocument(context.Background(), "doc-a", "a", false); err != nil {
		t.Fatalf("IndexDocument: %v", err)
	}

	result, err := indexer.Search(context.Background(), "b", true)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if result.AboveThreshold {
		t.Fatalf("expected below threshold search result: %+v", result)
	}
	if len(result.Centroid) == 0 {
		t.Fatalf("expected centroid when includeCentroid=true")
	}

	searchErrIndexer, err := NewIndexer(errEmbedder{err: errors.New("embed down")}, 1, 0.5)
	if err != nil {
		t.Fatalf("NewIndexer search err: %v", err)
	}
	_, err = searchErrIndexer.Search(context.Background(), "a", false)
	if err == nil || !strings.Contains(err.Error(), "embed document") {
		t.Fatalf("expected wrapped search embed error, got %v", err)
	}
}

func TestVectorHelpersHandleZeroAndMismatchedVectors(t *testing.T) {
	zero := []float64{0, 0}
	normalizeInPlace(zero)
	if zero[0] != 0 || zero[1] != 0 {
		t.Fatalf("zero vector should remain zero: %+v", zero)
	}
	if got := cosineSimilarityNormalized([]float64{1}, []float64{1, 0}, 1); got != 0 {
		t.Fatalf("mismatched similarity should be zero, got %f", got)
	}
	if got := cosineSimilarityNormalized([]float64{1}, []float64{1}, 0); got != 0 {
		t.Fatalf("zero norm similarity should be zero, got %f", got)
	}
}

func newTestIndexer(t *testing.T, categoryCount int, threshold float64) *Indexer {
	t.Helper()

	indexer, err := NewIndexer(fakeEmbedder{
		"a":  {1, 0},
		"a2": {1, 0},
		"b":  {0.8, 0.6},
		"c":  {0, 1},
	}, categoryCount, threshold)
	if err != nil {
		t.Fatalf("NewIndexer: %v", err)
	}
	return indexer
}

func assertFloatNear(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("float mismatch: got=%f want=%f", got, want)
	}
}
