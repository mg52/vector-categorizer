package vectorindex

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// fakeEmbedder maps text → vector and implements both Embedder and BatchEmbedder.
type fakeEmbedder map[string][]float64

func (f fakeEmbedder) Embed(_ context.Context, text string) ([]float64, error) {
	vec, ok := f[text]
	if !ok {
		return nil, fmt.Errorf("no embedding for %q", text)
	}
	return append([]float64(nil), vec...), nil
}

func (f fakeEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	vecs := make([][]float64, len(texts))
	for i, text := range texts {
		vec, err := f.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		vecs[i] = vec
	}
	return vecs, nil
}

// Orthogonal unit vectors for three test categories — similarity is unambiguous.
const (
	foodDesc   = "food, cooking, recipe"
	travelDesc = "travel, trip, journey"
	sportsDesc = "sports, exercise, fitness"
)

var testCategories = map[string]string{
	"food":   foodDesc,
	"travel": travelDesc,
	"sports": sportsDesc,
}

// testEmbedder maps category descriptions and query texts to known vectors.
var testEmbedder = fakeEmbedder{
	foodDesc:   {1, 0, 0},
	travelDesc: {0, 1, 0},
	sportsDesc: {0, 0, 1},
	// query texts
	"pasta recipe dinner": {0.9, 0.1, 0},
	"hotel booking Paris": {0.1, 0.9, 0},
	"gym workout running": {0, 0.05, 0.95},
}

func newTestCategorizer(t *testing.T) *Categorizer {
	t.Helper()
	c, err := NewCategorizer(context.Background(), testEmbedder, testCategories)
	if err != nil {
		t.Fatalf("NewCategorizer: %v", err)
	}
	return c
}

func absFloat(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func TestParseCategoriesValid(t *testing.T) {
	input := `
# This is a comment
food-recipe: food, lunch, dinner
travel: trip, journey

`
	cats, err := parseCategories(strings.NewReader(input))
	if err != nil {
		t.Fatalf("parseCategories: %v", err)
	}
	if len(cats) != 2 {
		t.Fatalf("expected 2 categories, got %d: %v", len(cats), cats)
	}
	if cats["food-recipe"] != "food, lunch, dinner" {
		t.Fatalf("food-recipe mismatch: %q", cats["food-recipe"])
	}
	if cats["travel"] != "trip, journey" {
		t.Fatalf("travel mismatch: %q", cats["travel"])
	}
}

func TestParseCategoriesDescriptionWithColon(t *testing.T) {
	// colons after the first one must be preserved in the description
	cats, err := parseCategories(strings.NewReader("url-category: see https://example.com for details\n"))
	if err != nil {
		t.Fatalf("parseCategories: %v", err)
	}
	if cats["url-category"] != "see https://example.com for details" {
		t.Fatalf("description mismatch: %q", cats["url-category"])
	}
}

func TestParseCategoriesErrors(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{name: "no categories (only comments)", input: "# only comment\n\n"},
		{name: "missing colon", input: "food no colon here\n"},
		{name: "empty description", input: "food:   \n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseCategories(strings.NewReader(tt.input))
			if err == nil {
				t.Fatalf("expected error for input %q", tt.input)
			}
		})
	}
}

func TestNewCategorizerValidation(t *testing.T) {
	ctx := context.Background()
	if _, err := NewCategorizer(ctx, nil, testCategories); err == nil {
		t.Fatalf("expected nil embedder error")
	}
	if _, err := NewCategorizer(ctx, testEmbedder, nil); err == nil {
		t.Fatalf("expected empty categories error")
	}
	if _, err := NewCategorizer(ctx, testEmbedder, map[string]string{}); err == nil {
		t.Fatalf("expected empty categories error")
	}
}

func TestNewCategorizerEmbedderError(t *testing.T) {
	emptyEmbedder := fakeEmbedder{}
	_, err := NewCategorizer(context.Background(), emptyEmbedder, map[string]string{"food": "food desc"})
	if err == nil || !strings.Contains(err.Error(), "embed categories") {
		t.Fatalf("expected embed categories error, got %v", err)
	}
}

func TestCategorizerCategorizeSelectsClosestCategory(t *testing.T) {
	c := newTestCategorizer(t)

	tests := []struct {
		text     string
		expected string
	}{
		{text: "pasta recipe dinner", expected: "food"},
		{text: "hotel booking Paris", expected: "travel"},
		{text: "gym workout running", expected: "sports"},
	}

	for _, tt := range tests {
		t.Run(tt.text, func(t *testing.T) {
			result, err := c.Categorize(context.Background(), tt.text)
			if err != nil {
				t.Fatalf("Categorize: %v", err)
			}
			if result.Category != tt.expected {
				t.Fatalf("category mismatch: got=%q want=%q", result.Category, tt.expected)
			}
			if result.Similarity <= 0 {
				t.Fatalf("expected positive similarity, got %f", result.Similarity)
			}
			if absFloat(result.Distance-(1-result.Similarity)) > 1e-9 {
				t.Fatalf("distance/similarity mismatch: sim=%f dist=%f", result.Similarity, result.Distance)
			}
		})
	}
}

func TestCategorizerCategorizeEmptyText(t *testing.T) {
	c := newTestCategorizer(t)
	_, err := c.Categorize(context.Background(), "")
	if err == nil || !strings.Contains(err.Error(), "text is required") {
		t.Fatalf("expected text required error, got %v", err)
	}
}

func TestCategorizerCategorizeEmbedderError(t *testing.T) {
	c := newTestCategorizer(t)
	_, err := c.Categorize(context.Background(), "this text has no embedding")
	if err == nil || !strings.Contains(err.Error(), "embed text") {
		t.Fatalf("expected embed text error, got %v", err)
	}
}

func TestCategorizerCategorizeEmptyEmbedding(t *testing.T) {
	embedder := fakeEmbedder{
		foodDesc:   {1, 0, 0},
		travelDesc: {0, 1, 0},
		sportsDesc: {0, 0, 1},
		"query":    {},
	}
	c, err := NewCategorizer(context.Background(), embedder, testCategories)
	if err != nil {
		t.Fatalf("NewCategorizer: %v", err)
	}
	_, err = c.Categorize(context.Background(), "query")
	if err == nil || !strings.Contains(err.Error(), "embedding is empty") {
		t.Fatalf("expected embedding is empty error, got %v", err)
	}
}

func TestCategorizerCategoryNames(t *testing.T) {
	c := newTestCategorizer(t)
	names := c.CategoryNames()
	if len(names) != len(testCategories) {
		t.Fatalf("expected %d category names, got %d: %v", len(testCategories), len(names), names)
	}
	nameSet := make(map[string]bool, len(names))
	for _, name := range names {
		nameSet[name] = true
	}
	for expected := range testCategories {
		if !nameSet[expected] {
			t.Fatalf("missing category %q in names: %v", expected, names)
		}
	}
}

func TestEmbedAllFallsBackToSingleEmbed(t *testing.T) {
	// singleEmbedder implements only Embedder (no EmbedBatch) to exercise the fallback path.
	type singleEmbedder struct{ fakeEmbedder }
	se := singleEmbedder{testEmbedder}

	vecs, err := embedAll(context.Background(), se, []string{foodDesc, travelDesc})
	if err != nil {
		t.Fatalf("embedAll: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(vecs))
	}
}

func TestVectorHelpers(t *testing.T) {
	// normalizeInPlace: zero vector stays zero
	zero := []float64{0, 0}
	normalizeInPlace(zero)
	if zero[0] != 0 || zero[1] != 0 {
		t.Fatalf("zero vector mutated: %v", zero)
	}

	// dotProduct: mismatched lengths return 0
	if got := dotProduct([]float64{1, 2}, []float64{1}); got != 0 {
		t.Fatalf("expected 0 for mismatched lengths, got %f", got)
	}

	// vectorNorm and normalizeInPlace round-trip
	vec := []float64{3, 4}
	normalizeInPlace(vec)
	norm := vectorNorm(vec)
	if absFloat(norm-1.0) > 1e-9 {
		t.Fatalf("expected unit norm after normalization, got %f", norm)
	}
}
