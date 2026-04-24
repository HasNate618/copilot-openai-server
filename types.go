package main

import (
	"encoding/json"
	"strings"
	"time"
)

// OpenAI API Request/Response Types

// ChatCompletionRequest represents an OpenAI chat completion request
type ChatCompletionRequest struct {
	Model               string         `json:"model"`
	Messages            []Message      `json:"messages"`
	Temperature         *float64       `json:"temperature,omitempty"`
	TopP                *float64       `json:"top_p,omitempty"`
	N                   *int           `json:"n,omitempty"`
	Stream              bool           `json:"stream"`
	StreamOptions       *StreamOptions `json:"stream_options,omitempty"`
	Stop                interface{}    `json:"stop,omitempty"`
	MaxCompletionTokens *int           `json:"max_completion_tokens,omitempty"`
	MaxTokens           *int           `json:"max_tokens,omitempty"`
	PresencePenalty     *float64       `json:"presence_penalty,omitempty"`
	FrequencyPenalty    *float64       `json:"frequency_penalty,omitempty"`
	ReasoningEffort     string         `json:"reasoning_effort,omitempty"`
	Tools               []Tool         `json:"tools,omitempty"`
	ToolChoice          interface{}    `json:"tool_choice,omitempty"`
	User                string         `json:"user,omitempty"`
}

// StreamOptions represents chat-completions streaming options.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role    string         `json:"role,omitempty"`
	Content MessageContent `json:"content,omitempty"`
	// Reasoning carries model reasoning text when provided by the Copilot SDK
	// (assistant.reasoning / assistant.reasoning_delta). Optional.
	Reasoning  string     `json:"reasoning,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// MessageContent supports both string and structured OpenAI content arrays.
type MessageContent struct {
	Text  string        `json:"-"`
	Parts []ContentPart `json:"-"`
}

// String returns a plain-text representation of the content.
func (c MessageContent) String() string {
	return c.Text
}

// MarshalJSON serializes content back to a standard string payload.
func (c MessageContent) MarshalJSON() ([]byte, error) {
	return json.Marshal(c.Text)
}

// UnmarshalJSON accepts either a plain string, null, or a content-parts array.
func (c *MessageContent) UnmarshalJSON(data []byte) error {
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "" || trimmed == "null" {
		c.Text = ""
		return nil
	}

	if trimmed[0] == '"' {
		return json.Unmarshal(data, &c.Text)
	}

	var parts []ContentPart
	if err := json.Unmarshal(data, &parts); err != nil {
		return err
	}

	var builder strings.Builder
	for _, part := range parts {
		text := part.PlainText()
		if text == "" {
			continue
		}
		if builder.Len() > 0 {
			builder.WriteString("\n")
		}
		builder.WriteString(text)
	}

	c.Text = builder.String()
	c.Parts = parts // retain structured parts so callers can create attachments
	return nil
}

// ContentPart represents a structured OpenAI content part.
type ContentPart struct {
	Type       string          `json:"type"`
	Text       string          `json:"text,omitempty"`
	Refusal    string          `json:"refusal,omitempty"`
	ImageURL   *ImageURLPart   `json:"image_url,omitempty"`
	InputAudio *InputAudioPart `json:"input_audio,omitempty"`
	File       *FilePart       `json:"file,omitempty"`
}

// PlainText flattens supported content parts into prompt text.
func (p ContentPart) PlainText() string {
	switch p.Type {
	case "text":
		return p.Text
	case "refusal":
		return p.Refusal
	case "image_url":
		if p.ImageURL != nil && p.ImageURL.URL != "" {
			return "[Image: " + p.ImageURL.URL + "]"
		}
	case "input_audio":
		if p.InputAudio != nil {
			return "[Audio input]"
		}
	case "file":
		if p.File != nil {
			if p.File.Filename != "" {
				return "[File: " + p.File.Filename + "]"
			}
			if p.File.FileID != "" {
				return "[File: " + p.File.FileID + "]"
			}
			return "[File attachment]"
		}
	}

	return ""
}

// ImageURLPart represents an image_url content payload.
type ImageURLPart struct {
	URL string `json:"url"`
}

// InputAudioPart represents an input_audio payload.
type InputAudioPart struct {
	Data   string `json:"data,omitempty"`
	Format string `json:"format,omitempty"`
}

// FilePart represents a file content payload.
type FilePart struct {
	FileID   string `json:"file_id,omitempty"`
	FileData string `json:"file_data,omitempty"`
	Filename string `json:"filename,omitempty"`
}

// Tool represents a tool definition
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction represents a function definition within a tool
type ToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ToolCall represents a tool call made by the assistant
type ToolCall struct {
	Index    *int             `json:"index,omitempty"` // Required for streaming tool calls
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction represents the function details in a tool call
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatCompletionResponse represents an OpenAI chat completion response
type ChatCompletionResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	Usage             *Usage   `json:"usage,omitempty"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
}

// Choice represents a completion choice
type Choice struct {
	Index        int      `json:"index"`
	Message      *Message `json:"message,omitempty"`
	Delta        *Message `json:"delta,omitempty"`
	FinishReason *string  `json:"finish_reason"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionChunk represents a streaming chunk
type ChatCompletionChunk struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
}

// ModelsResponse represents the response for /v1/models
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelData `json:"data"`
}

// ModelData represents a single model in the models list
type ModelData struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ErrorResponse represents an API error
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains error details
type ErrorDetail struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param,omitempty"`
	Code    *string `json:"code,omitempty"`
}

// Helper function to get current timestamp
func currentTimestamp() int64 {
	return time.Now().Unix()
}

// Helper to create a string pointer
func strPtr(s string) *string {
	return &s
}
