# Social Media AI Agent

A Flask-based Arabic language virtual assistant that integrates with Facebook Messenger and page comments, featuring speech recognition, context-aware responses, and natural language generation.

> **IMPORTANT NOTE:** To activate the audio interface of this application, please contact the student developers to activate an ngrok server in order for the webhook to be accessible for public use.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Components](#components)
  - [Facebook Integration](#facebook-integration)
  - [Natural Language Processing](#natural-language-processing)
  - [Audio Processing](#audio-processing)
  - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
- [Workflows](#workflows)
  - [Text Message Handling](#text-message-handling)
  - [Voice Message Handling](#voice-message-handling)
  - [Comment Handling](#comment-handling)
- [Setup and Installation](#setup-and-installation)
- [Environment Variables](#environment-variables)
- [API References](#api-references)
- [Dependencies](#dependencies)

## Overview

This application provides an Arabic language assistant that interfaces with users via Facebook Messenger and page comments. It uses AI models for natural language understanding and generation, with context-aware responses powered by a hybrid retrieval system.

## Features

- ✅ Facebook Messenger integration
- ✅ Facebook page comment monitoring and responses
- ✅ Arabic speech recognition (ASR)
- ✅ Text-to-speech for Arabic with proper diacritics
- ✅ Retrieval-augmented generation for context-aware responses
- ✅ Hybrid search combining BM25 and semantic approaches
- ✅ Intelligent response formatting for Arabic text

## System Architecture

```
┌─────────────────┐    ┌───────────────────┐    ┌─────────────────────┐
│ Facebook API    │    │ Flask Application │    │ Audio Processing    │
│ - Messenger     │◄──►│ - Webhook handler │◄──►│ - Speech recognition│
│ - Page Comments │    │ - Message routing │    │ - Text-to-Speech    │
└─────────────────┘    └───────────────────┘    └─────────────────────┘
         ▲                       ▲                        ▲
         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 │
                   ┌─────────────▼────────────┐
                   │ AI & Retrieval Systems   │
                   │ - Gemini LLM             │
                   │ - RAG system             │
                   │ - BM25 search            │
                   │ - Semantic search        │
                   └──────────────────────────┘
```

## Components

### Facebook Integration

The application connects to Facebook through:

- **Webhook Handler**: Verifies and processes incoming Facebook webhook events
- **Messenger Bot**: Processes and responds to user messages from Facebook Messenger
- **Comment Bot**: Monitors Facebook page posts and responds to comments

Key functions:
- `webhook()`: Handles incoming webhook events
- `run_messenger_bot()`: Polls for new messages from users
- `run_comment_bot()`: Polls for new comments on posts
- `send_message()`: Delivers text responses to users
- `send_audio_to_user()`: Delivers audio responses to users

### Natural Language Processing

Text processing capabilities include:

- **Response Generation**: Uses Google's Gemini LLM to generate high-quality Arabic responses
- **Arabic Text Processing**: Formats and structures Arabic text for natural reading
- **Diacritization**: Adds proper Arabic diacritics (tashkeel) for text-to-speech

Key functions:
- `get_gemini_response()`: Generates responses using Gemini LLM
- `split_arabic_paragraph()`: Splits long Arabic text into manageable sections
- `doTashkeel()`: Adds diacritical marks to Arabic text

### Audio Processing

Audio handling capabilities include:

- **Speech Recognition**: Converts Arabic speech to text using Whisper ASR
- **Text-to-Speech**: Converts text responses to natural-sounding Arabic speech
- **Audio Delivery**: Manages uploading and sending audio files to users

Key functions:
- `process_audio()`: Handles audio message processing pipeline
- `tts()`: Converts text to speech with Arabic support
- `upload_audio_attachment()`: Uploads audio files to Facebook
- `send_audio_message_with_attachment()`: Sends audio to users via Messenger

### Retrieval-Augmented Generation (RAG)

The application implements a sophisticated knowledge retrieval system:

- **Document Processing**: Chunks and preprocesses text documents
- **Classical Search**: Uses BM25 algorithm for keyword-based retrieval
- **Semantic Search**: Employs FAISS and sentence embeddings for meaning-based retrieval
- **Hybrid Approach**: Combines both search methods for optimal results

Key functions:
- `load_and_chunk_text()`: Processes and chunks text documents
- `setup_classical_search()`: Configures BM25 search index
- `semantic_search()`: Performs embedding-based semantic search
- `hybrid_retrieve()`: Combines search approaches for best results

## Workflows

### Text Message Handling

1. Webhook receives a text message from Facebook
2. `process_message()` extracts the message text
3. `hybrid_retrieve()` finds relevant context from the knowledge base
4. `get_gemini_response()` generates a response using the LLM
5. `split_arabic_paragraph()` divides the response into manageable sections
6. `send_message()` delivers each section to the user

### Voice Message Handling

1. Webhook receives an audio message from Facebook
2. `process_audio()` downloads the audio file
3. Whisper ASR transcribes audio to text
4. `get_gemini_response()` generates a response
5. `doTashkeel()` adds diacritics to the response text
6. `tts()` converts the text to speech
7. `send_audio_to_user()` delivers the audio response

### Comment Handling

1. `run_comment_bot()` polls for new comments on page posts
2. `get_all_threads()` retrieves comment threads
3. `get_latest_comment()` identifies the most recent comment
4. `get_gemini_response()` generates an appropriate reply
5. `send_reply()` posts the response as a comment reply

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables (see below)
4. Run the application:
   ```
   python app.py
   ```

## Environment Variables

Create a `.env` file with the following variables:

```
GEMINI_API_KEY=your_gemini_api_key
PAGE_ACCESS_TOKEN=your_facebook_page_access_token
VERIFY_TOKEN=your_webhook_verification_token
```

## API References

- **Facebook Graph API**: Used for Messenger and page interactions
- **Google Generative AI**: Used for the Gemini LLM
- **Whisper ASR**: Used for speech recognition
- **FAISS**: Used for efficient similarity search
- **SentenceTransformer**: Used for text embeddings

## Dependencies

Major dependencies include:

- **Flask**: Web framework for handling requests
- **Whisper**: OpenAI's speech recognition model
- **Google Generative AI**: Gemini LLM for text generation
- **FAISS**: Facebook AI Similarity Search for efficient vector search
- **SentenceTransformer**: For generating text embeddings
- **BM25Okapi**: For classical text search
- **NLTK**: For natural language processing utilities
- **NumPy**: For numerical operations
- **Requests**: For HTTP requests
