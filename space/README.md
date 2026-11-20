---
title: Callie
emoji: 📞
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Call an AI support agent for a bookstore by browser or phone
---

# Callie: the bookstore's AI phone line

Press **Record**, allow the microphone, and talk to the bookstore's phone assistant, or call the Twilio
number if one is shown in the project's README.

**Demo customer:** order **5512**, last name **Virk**. Try *"Where is my order?"*, *"The book arrived
damaged, please open a ticket"*, *"Do you have any books by Erik Gross?"*, or someone else's order.

- **The call:** phone audio (8 kHz μ-law) → Whisper → a fixed verification script → an AI agent with tools
  over two MCP servers → Piper's voice.
- **Safety:** verification before anything, only the caller's own data (enforced by the Support Desk), tickets
  only after the caller says yes, a TalkTable that can see only the book catalog, rate limits.
- **Demo data:** fictional, and it resets whenever this Space restarts. The Space sleeps when unused, so the
  first visit can take a minute.

Code and design: [github.com/grishika21/talktable](https://github.com/grishika21/talktable)
