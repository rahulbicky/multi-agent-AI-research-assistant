# docs/screenshots/

This directory stores real application screenshots for the README.

## Required Screenshots

Please capture these by opening the app at http://localhost:8501:

| Filename | What to capture |
|:---|:---|
| `01_welcome_screen.png` | Hero header + 4 capability cards (empty chat state) |
| `02_chat_response.png` | A completed AI exchange with agent badge visible |
| `03_doubt_window.png` | The Contextual Doubt Window open in the right panel |
| `04_document_upload.png` | Sidebar with an uploaded document listed |
| `05_rag_citations.png` | AI response with document source citations |
| `06_web_research.png` | Research agent response with Sources section |
| `07_conversation_list.png` | Sidebar with several named conversations |
| `08_light_theme.png` | The full UI in Light theme |
| `09_coding_response.png` | A code block response from the Coding agent |

## Tips for Good Screenshots

- Use a 1280×800 or 1440×900 browser window for consistency
- Use the dark theme for most screenshots (the default)
- Capture the Light theme screenshot after toggling the theme button
- Crop to remove your browser chrome (tabs, address bar)
- Save as PNG for best quality; aim for files under 300 KB each
- Use Snipping Tool (Win + Shift + S) on Windows

## After capturing

Replace the placeholder section in README.md with actual image tags:

```markdown
![Welcome Screen](docs/screenshots/01_welcome_screen.png)
*Welcome screen with capability cards*
```
