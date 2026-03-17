package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestBuildPrompt(t *testing.T) {
	tests := []struct {
		name     string
		messages []Message
		want     string
		wants    []string // Substrings we expect
		ignores  []string // Substrings we expect NOT to appear
	}{
		{
			name: "Basic user assistant interaction",
			messages: []Message{
				{Role: "user", Content: MessageContent{Text: "Hello"}},
				{Role: "assistant", Content: MessageContent{Text: "Hi there"}},
			},
			wants: []string{
				"[User]: Hello",
				"[Assistant]: Hi there",
			},
		},
		{
			name: "System message should be ignored in prompt",
			messages: []Message{
				{Role: "system", Content: MessageContent{Text: "You are a helpful assistant"}},
				{Role: "user", Content: MessageContent{Text: "Hello"}},
			},
			wants: []string{
				"[User]: Hello",
			},
			ignores: []string{
				"[System]: You are a helpful assistant",
				"You are a helpful assistant",
			},
		},
		{
			name: "Multiple system messages should be ignored",
			messages: []Message{
				{Role: "system", Content: MessageContent{Text: "Sys 1"}},
				{Role: "user", Content: MessageContent{Text: "User 1"}},
				{Role: "system", Content: MessageContent{Text: "Sys 2"}},
			},
			wants: []string{
				"[User]: User 1",
			},
			ignores: []string{
				"Sys 1",
				"Sys 2",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildPrompt(tt.messages)

			for _, want := range tt.wants {
				if !strings.Contains(got, want) {
					t.Errorf("buildPrompt() missing expected content %q. Got:\n%s", want, got)
				}
			}

			for _, ignore := range tt.ignores {
				if strings.Contains(got, ignore) {
					t.Errorf("buildPrompt() contained prohibited content %q. Got:\n%s", ignore, got)
				}
			}
		})
	}
}

func TestChatCompletionRequestSupportsStructuredContent(t *testing.T) {
	body := []byte(`{
		"model": "gpt-5",
		"reasoning_effort": "high",
		"tool_choice": {"type": "function", "function": {"name": "lookup_issue"}},
		"messages": [
			{"role": "developer", "content": [{"type": "text", "text": "Follow repository rules."}]},
			{"role": "user", "content": [{"type": "text", "text": "Summarize this file."}, {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}]},
			{"role": "tool", "tool_call_id": "call_123", "content": [{"type": "text", "text": "Issue loaded."}]}
		],
		"tools": [{
			"type": "function",
			"function": {
				"name": "lookup_issue",
				"parameters": {"type": "object"}
			}
		}]
	}`)

	var req ChatCompletionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if got := req.Messages[0].Content.String(); got != "Follow repository rules." {
		t.Fatalf("developer content = %q, want %q", got, "Follow repository rules.")
	}

	if got := req.Messages[1].Content.String(); got != "Summarize this file.\n[Image: https://example.com/image.png]" {
		t.Fatalf("user content = %q", got)
	}

	if got := req.Messages[2].Content.String(); got != "Issue loaded." {
		t.Fatalf("tool content = %q, want %q", got, "Issue loaded.")
	}

	if req.ReasoningEffort != "high" {
		t.Fatalf("reasoning effort = %q, want %q", req.ReasoningEffort, "high")
	}
}

func TestDetermineAvailableToolsHonorsToolChoice(t *testing.T) {
	tools := []Tool{
		{Type: "function", Function: ToolFunction{Name: "lookup_issue"}},
		{Type: "function", Function: ToolFunction{Name: "lookup_pr"}},
	}

	selected := determineAvailableTools(tools, map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name": "lookup_pr",
		},
	})

	if len(selected) != 1 || selected[0] != "lookup_pr" {
		t.Fatalf("selected tools = %v, want [lookup_pr]", selected)
	}

	none := determineAvailableTools(tools, "none")
	if len(none) != 0 {
		t.Fatalf("selected tools for none = %v, want empty", none)
	}
}
