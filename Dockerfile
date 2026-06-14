FROM golang:1.25-alpine AS builder
WORKDIR /app
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o vector-categorizer ./cmd

FROM alpine:3.21
RUN apk --no-cache add ca-certificates
WORKDIR /app
COPY --from=builder /app/vector-categorizer .

ENV VECTOR_INDEX_ADDR=:8090
ENV VECTOR_INDEX_CATEGORY_COUNT=100
ENV VECTOR_INDEX_SIMILARITY_THRESHOLD=0.60
ENV VECTOR_INDEX_EMBED_URL=http://localhost:11434/api/embed
ENV VECTOR_INDEX_EMBED_MODEL=nomic-embed-text

EXPOSE 8090
ENTRYPOINT ["./vector-categorizer"]
