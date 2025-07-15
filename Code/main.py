from flask import Flask, request
import os
import requests
import threading
import whisper
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from tts_arabic import tts
import time
from multiprocessing import Process
import os
import re
import sys
import threading
import pickle
import numpy as np
import faiss
import tkinter as tk
from tkinter import scrolledtext
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()  # take environment variables from .env

processed_messages = []
# Dictionary to store last message ID per user
user_last_message_ids = {}

import re

# Checks whether to respond and sends message if needed
def process_message(sender_id, message_id, mssg_txt):
    if sender_id != PAGE_ID:
        last_seen_id = user_last_message_ids.get(sender_id)
        if last_seen_id != message_id:
            wordLimit = 1000
            sentenceLimit = 20
            context = hybrid_retrieve(mssg_txt, emb_model, faiss_index, chunks)
            prompt = f"""
            سيتم تزويدك بسياق إضافي على شكل مصفوفة تحتوي على معلومات يمكنك استخدامها لدعم إجابتك.

            إذا كان النص المرسل عبارة عن تحية أو تعليق عام لا يتضمن سؤالاً أو استفساراً واضحاً، فأجب بشكل طبيعي ومباشر دون الرجوع إلى السياق.

            أما إذا كان النص يحتوي على سؤال أو استفسار، فافحص ما إذا كان مرتبطاً بموضوع السياق. 
            - إن كان مرتبطاً، فاستخدم المعلومات المتوفرة في السياق لدعم إجابتك.
            - وإن لم يكن مرتبطاً بالسياق، فاشرح أن السؤال خارج نطاق محتوى الصفحة أو الموضوع المطروح.

            اكتب الرد بلغة عربية فصحى واضحة وسلسة في حدود {wordLimit} كلمة، بحيث تكون الإجابة فقرة متصلة تتكون من جمل قصيرة لا يتجاوز طول كل جملة {sentenceLimit} كلمة. 
            لا حاجة لبلوغ الحد الأقصى لعدد الكلمات إذا لم يتطلب المعنى ذلك، فالإيجاز مرحب به ما دام يحقق الوضوح. تجنب استخدام الرموز أو التعداد أو أي علامات غير مألوفة، ويُفضّل أن تكون الإجابة قابلة للقراءة بصوت عالٍ دون انقطاع أو تعقيد.

            هذا هو النص: ({mssg_txt})
            وهذا هو السياق: {context}
            """
            # prompt = f"سيتم تزويدك بسياق إضافي على شكل مصفوفة تحتوي على معلومات يمكنك استخدامها لدعم إجابتك. اكتب رداً على النص التالي بلغة عربية فصحى واضحة وسلسة، في حدود {wordLimit} كلمة، بحيث تكون الإجابة فقرة متصلة مكونة من جمل قصيرة لا يتجاوز طول كل جملة {sentenceLimit} كلمة. لا حاجة للوصول إلى الحد الأقصى لعدد الكلمات إذا لم يتطلب المعنى ذلك، فالإيجاز مرحب به ما دام يحقق الوضوح. تجنب استخدام الرموز أو التعداد أو أي علامات غير معتادة، وحرصاً على الانسيابية يجب أن تكون الإجابة قابلة للقراءة بصوت عالٍ بشكل طبيعي دون توقف أو تعقيد: ({mssg_txt}). هذا هو السياق: {context}"
            # prompt = f"اكتب رداً على النص التالي بلغة عربية فصحى واضحة وسلسة، على شكل فقرة متصلة مكونة من جمل قصيرة لا يتجاوز طول كل جملة {sentenceLimit} كلمة. لا حاجة للوصول إلى الحد الأقصى لعدد الكلمات ({wordLimit}) إذا لم يتطلب المعنى ذلك، فالإيجاز مرحب به ما دام يحقق الوضوح دون تكرار لنفس الكلام. تجنب استخدام الرموز أو التعداد أو أي علامات غير معتادة، وحرصاً على الانسيابية يجب أن تكون الإجابة قابلة للقراءة بصوت عالٍ بشكل طبيعي دون توقف أو تعقيد: {mssg_txt}"
            reply = get_gemini_response(gemini_model, prompt)
            
            sections = split_arabic_paragraph(reply)
            print(f"LLM response split successfully into {len(sections)} sections.")

            for i in range(len(sections)):
                print(f"Sending section {i+1}/{len(sections)}")
                send_message(sender_id, sections[i])

            user_last_message_ids[sender_id] = message_id
        else:
            logging.info(f"No new message from user {sender_id}")
    else:
        logging.info("Last message was from the page, skipping.")

def split_arabic_paragraph(paragraph):
    # Split the paragraph into sentences using common Arabic punctuation
    sentence_endings = re.compile(r'(?<=[.؟!])\s+')
    sentences = sentence_endings.split(paragraph.strip())
    
    # Group every two sentences into a section
    sections = []
    for i in range(0, len(sentences), 4):
        group = ' '.join(sentences[i:i+4]).strip()
        if group:
            sections.append(group)
    
    return sections

def setup_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

def get_gemini_response(model, prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting response: {str(e)}"
    
# Set up Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_model = setup_gemini(gemini_api_key)

VERIFY_TOKEN = "my_secret_token"
# Your Page Access Token
PAGE_ACCESS_TOKEN = "EAAjG53j9HfEBO29zN4dLDEXmX9xhz6ZAyPS7Daks7VCyZBqXp2wihyNKVXX4PoqN1vAXSpXDSZBePefDwlQw1ZCV23B5iBsq1ZA3hbwZBIAvGnDVq5TGkCMnXonKb7NvI6D57VHtBZB5R7fVkDx8gSgWrkSsXrUXB7mJxl4VntO1m8QFz6dlKjEkZBSWDXDd"
# PAGE_ACCESS_TOKEN = "EAAjG53j9HfEBOy6j8OQ5mC9ZC3sk0CLgOMuCXoeX3BrbhVYLvgntrQrbZAuqmZAdJ2IOZAIuCpmgBZBcLqZBzo5Jp5Sk3R3rBsiiBo7qPGsuXLBUKICQjCaZAeNpJcYcWxE6f6OkOBcgZByAGxKGzd3ZAQ3iYqLwZADFFWlxMjOIYdYMZCmxVOS0junkrcXOcofRSfZA"
PAGE_ID = '666846913177212'
GRAPH_VERSION = 'v22.0'
GRAPH_URL = f'https://graph.facebook.com/{GRAPH_VERSION}'

asr_model = whisper.load_model("turbo")

def doTashkeel(model, query):
    prompt = f"ضع التشكيل المناسب على كل حرف في هذه الجملة: ({query})"

    return get_gemini_response(model, prompt)

# Sends a message to a user
def send_message(user_id, text):
    url = "https://graph.facebook.com/v22.0/me/messages"
    params = {'access_token': PAGE_ACCESS_TOKEN}
    headers = {'Content-Type': 'application/json'}
    data = {
        'messaging_type': 'RESPONSE',
        'recipient': {'id': user_id},
        'message': {'text': text}
    }

    response = requests.post(url, params=params, headers=headers, json=data)
    if response.status_code == 200:
        logging.info(f"Sent message to {user_id}")
    else:
        logging.error(f"Failed to send message to {user_id}: {response.status_code} - {response.text}")

def process_audio(audio_url, sender_id):
    print(f"There is an audio from {sender_id}")
    response = requests.get(audio_url, headers={"Authorization": f"Bearer {PAGE_ACCESS_TOKEN}"})
    if response.status_code == 200:
        with open("received_message.mp4", "wb") as f:
            f.write(response.content)
        print("Audio downloaded successfully.")
        
        result = asr_model.transcribe("received_message.mp4", language="ar")
        print("Audio transcribed successfully.")
        wordLimit = 250
        sentenceLimit = 20
        prompt = f"اكتب رداً على النص التالي بلغة عربية واضحة وسلسة في حدود {wordLimit} كلمة، بحيث تكون الإجابة فقرة متصلة مكونة من جمل قصيرة لا يتجاوز طول كل جملة {sentenceLimit} كلمة. تجنب استخدام أي رموز أو تنسيقات غير معتادة، ولا تستخدم التعداد أو التنقيط. يجب أن تكون الإجابة قابلة للقراءة بصوت عالٍ بشكل طبيعي دون توقف أو تعقيد: {result["text"]}"
        reply = get_gemini_response(gemini_model, prompt)
        # reply = get_gemini_response(gemini_model, result["text"])
        with open("llm_audio_response.txt", "w", encoding="utf-8") as f:
            f.write(reply)
        print("LLM response received successfully.")
        sections = split_arabic_paragraph(reply)
        print(f"LLM response split successfully into {len(sections)} sections.")

        for i in range(len(sections)):
            print(f"Converting Section {i} to speech")

            ttsInput = doTashkeel(gemini_model, sections[i])

            with open("tts_input.txt", "w", encoding="utf-8") as f:
                f.write(ttsInput)
            
            wave = tts(
                ttsInput, # input text
                speaker = 1, # speaker id; choose between 0,1,2,3
                pace = 1, # speaker pace
                denoise = 0.005, # vocoder denoiser strength
                volume = 0.9, # Max amplitude (between 0 and 1)
                # play = True, # play audio?
                pitch_mul = 1, # pitch multiplier
                pitch_add = 0, # pitch offset
                vowelizer = None, # vowelizer model
                model_id = 'fastpitch', # Model ID for Text->Mel model
                vocoder_id = 'hifigan', # Model ID for vocoder model
                cuda = None, # Optional; CUDA device index
                save_to = './output.wav', # Optionally; save audio WAV file
                bits_per_sample = 32, # when save_to is specified (8, 16 or 32 bits)
                )
            print(f"Converted Section {i} to speech")
            print(f"Sending Section {i} as speech")
            success = send_audio_to_user("output.wav", sender_id)
            if success:
                logging.info("Audio sent successfully.")
            else:
                logging.error("Failed to send audio.")
    else:
        print(f"Failed to download audio: {response.status_code}\n{response.text}")

def process_text(received_message, sender_id):
    print(f"There is a text from {sender_id}")
        
    reply = get_gemini_response(gemini_model, received_message)
    print("LLM response received successfully.")
    sections = split_arabic_paragraph(reply)
    print(f"LLM response split successfully into {len(sections)} sections.")

    for i in range(len(sections)):
        print(f"Sending Section {i}")
        send_message(sender_id, sections[i])

# ========== Upload Audio File ==========
def upload_audio_attachment(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None

    url = f"{GRAPH_URL}/me/message_attachments"
    params = {'access_token': PAGE_ACCESS_TOKEN}
    files = {
        'filedata': (file_path, open(file_path, 'rb'), 'audio/wav')
    }
    data = {
        'message': '{"attachment":{"type":"audio", "payload":{"is_reusable":true}}}'
    }

    try:
        response = requests.post(url, params=params, data=data, files=files)
        if response.status_code == 200:
            attachment_id = response.json()['attachment_id']
            logging.info(f"Uploaded audio. Attachment ID: {attachment_id}")
            return attachment_id
        else:
            logging.error(f"Failed to upload audio: {response.status_code} - {response.text}")
            return None
    finally:
        files['filedata'][1].close()


# ========== Send Audio by Attachment ID ==========
def send_audio_message_with_attachment(recipient_id, attachment_id):
    url = f"{GRAPH_URL}/me/messages"
    params = {'access_token': PAGE_ACCESS_TOKEN}
    headers = {'Content-Type': 'application/json'}

    payload = {
        'recipient': {'id': recipient_id},
        'message': {
            'attachment': {
                'type': 'audio',
                'payload': {
                    'attachment_id': attachment_id
                }
            }
        }
    }

    response = requests.post(url, params=params, json=payload, headers=headers)
    if response.status_code == 200:
        logging.info(f"Audio message sent to {recipient_id}")
        return True
    else:
        logging.error(f"Failed to send audio: {response.status_code} - {response.text}")
        return False
    
# ========== PUBLIC FUNCTION ==========
def send_audio_to_user(audio_path, user_id):
    """
    Uploads a local audio file and sends it to a specific user via Messenger.
    :param audio_path: Path to the local audio file
    :param user_id: Facebook PSID of the user
    """
    attachment_id = upload_audio_attachment(audio_path)
    if attachment_id:
        return send_audio_message_with_attachment(user_id, attachment_id)
    else:
        logging.error("Audio upload failed. Message not sent.")
        return False

# Retrieve recent conversations
def get_all_conversations():
    conversations = []
    url = f"https://graph.facebook.com/v15.0/{PAGE_ID}/conversations"
    params = {
        'access_token': PAGE_ACCESS_TOKEN,
        'fields': 'id,messages.limit(1){id,message,from,created_time}',
        'limit': 50
    }

    while url:
        print("Enter Loop")
        response = requests.get(url, params=params)
        print("Got Response")
        if response.status_code != 200:
            logging.error(f"Failed to get conversations: {response.status_code} - {response.text}")
            break

        data = response.json()
        conversations.extend(data.get('data', []))
        url = data.get('paging', {}).get('next', None)
        params = None

    return conversations

def poll_new_messages(interval=10):
    while True:
        run_comment_bot()
        run_messenger_bot()

        time.sleep(interval)

# Main polling loop
def run_messenger_bot(interval=5):
    print("Start Messenger Bot")
    try:
        logging.info("Checking for new user messages...")
        conversations = get_all_conversations()

        for convo in conversations:
            messages = convo.get('messages', {}).get('data', [])
            if messages:
                last_msg = messages[0]
                mssg_txt = last_msg["message"]
                if mssg_txt:
                    print(mssg_txt)
                    sender_id = last_msg['from']['id']
                    message_id = last_msg['id']
                    process_message(sender_id, message_id, mssg_txt)

    except Exception as e:
        logging.error(f"Error in loop: {str(e)}")

###########################
##########################

# Track comment IDs we've already replied to
replied_to_comment_ids = set()


def get_page_posts():
    url = f"https://graph.facebook.com/v15.0/{PAGE_ID}/posts"
    params = {
        'access_token': PAGE_ACCESS_TOKEN,
        'fields': 'id',
        'limit': 10
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return [post['id'] for post in response.json().get('data', [])]
    else:
        logging.error(f"Error fetching posts: {response.text}")
        return []


def fetch_replies(comment_id):
    """Recursively fetch all replies and sub-replies for a given comment."""
    replies = []
    url = f"https://graph.facebook.com/v15.0/{comment_id}/comments"
    params = {
        'access_token': PAGE_ACCESS_TOKEN,
        'fields': 'id,message,from,created_time',
        'limit': 100
    }

    while url:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logging.error(f"Failed to fetch replies: {response.status_code} - {response.text}")
            break

        data = response.json()
        for reply in data.get('data', []):
            replies.append(reply)
            # Recursively get sub-replies
            sub_replies = fetch_replies(reply['id'])
            replies.extend(sub_replies)

        url = data.get('paging', {}).get('next')
        params = None

    return replies


def get_all_threads(post_id):
    """Get all top-level comments and all nested replies under each comment."""
    threads = []
    url = f"https://graph.facebook.com/v15.0/{post_id}/comments"
    params = {
        'access_token': PAGE_ACCESS_TOKEN,
        'fields': 'id,message,from,created_time',
        'limit': 100
    }

    while url:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logging.error(f"Failed to fetch top-level comments: {response.status_code} - {response.text}")
            break

        data = response.json()
        for comment in data.get('data', []):
            thread = [comment]
            replies = fetch_replies(comment['id'])
            thread.extend(replies)
            threads.append(thread)

        url = data.get('paging', {}).get('next')
        params = None

    return threads


def get_latest_comment(thread):
    """Returns the latest comment in the thread by created_time."""
    return max(thread, key=lambda x: x.get("created_time", ""))


def send_reply(comment_id, text):
    url = f"https://graph.facebook.com/v15.0/{comment_id}/comments"
    params = {'access_token': PAGE_ACCESS_TOKEN}
    data = {'message': text}
    response = requests.post(url, params=params, json=data)
    if response.status_code == 200:
        logging.info(f"Replied to comment {comment_id}")
    else:
        logging.error(f"Failed to reply: {response.status_code} - {response.text}")


def run_comment_bot(poll_interval=15):
    print("Starting comment bot...")
    try:
        print("Retrieving Posts.....")
        posts = get_page_posts()
        print("Retrieved Posts.....")
        for post_id in posts:
            print(f"Retrieving Threads for {post_id}.....")
            threads = get_all_threads(post_id)
            print(f"Retrieved Threads for {post_id}.....")

            for thread in threads:
                print(f"Retrieving Latest Comment.....")
                latest = get_latest_comment(thread)
                print(f"Retrieved Latest Comment.....")
                print(latest)
                comment_id = latest['id']
                if 'from' not in latest:
                    sender_id = 'blabla'
                else:
                    sender_id = latest['from']['id']

                if comment_id not in replied_to_comment_ids:
                    if sender_id != PAGE_ID:
                        wordLimit = 100
                        sentenceLimit = 20
                        prompt = f"رد على هذا الكلام فيما لا يزيد عن {wordLimit} كلمة بلغة عربية واضحة وسلسة، على شكل فقرة نصية مكونة من مجموعة من الجمل التي لا تتعدى كل واحدة منها {sentenceLimit} كلمة دون استخدام رموز أو نقاط أو علامات غير معتادة. تجنب التنقيط أو التعداد، وقدم الإجابة كسرد متصل وطبيعي يمكن قراءته بصوت عالٍ دون انقطاع: {latest["message"]}"
                        reply = get_gemini_response(gemini_model, prompt)
                        send_reply(comment_id, reply)
                        replied_to_comment_ids.add(comment_id)
                    else:
                        print(f"Last comment in thread is from the page — no reply sent.")


    except Exception as e:
        logging.error(f"Error in comment bot loop: {str(e)}")

###########################
##########################

@app.route("/", methods=["GET", "POST"])
def webhook():
    print("Test Test Test =====")
    if request.method == "GET":
        # Webhook verification
        if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.verify_token") == VERIFY_TOKEN:
            print("Test Test Test =====")
            return request.args.get("hub.challenge")
        return "Verification token mismatch", 403

    if request.method == "POST":
        data = request.get_json()
        print("Webhook received:", data)

        if "message" in data["entry"][0]["messaging"][0]:
            if data["entry"][0]["messaging"][0]["message"]["mid"] in processed_messages:
                return "OK", 200
            processed_messages.append(data["entry"][0]["messaging"][0]["message"]["mid"])
            sender_id = data['entry'][0]['messaging'][0]['sender']['id']

            if sender_id != PAGE_ID:
                if "text" in data["entry"][0]["messaging"][0]["message"]:
                    return "OK", 200 
                if "attachments" in data["entry"][0]["messaging"][0]["message"]:
                    audio_url = data["entry"][0]["messaging"][0]["message"]["attachments"][0]["payload"]["url"]
                    threading.Thread(target=process_audio, args=(audio_url,sender_id,)).start()

        # You can now inspect `data` for voice messages
        return "OK", 200

# Download Arabic stopwords
nltk.download('stopwords', quiet=True)
arabic_stopwords = set(stopwords.words('arabic'))

# ----------------------------
# STEP 0: Cleaning Functions
# ----------------------------

def clean_arabic_text_gui(text):
    # For general GUI cleaning
    pattern = r"[^؀-ۿݐ-ݿࢠ-ࣿ0-9a-zA-Z٠-٩\s.,!?؟،؛\-\n]"
    return re.sub(pattern, '', text)

def normalize_arabic(text):
    # For classical search normalization
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\w\s.]', '', text)  # Keep fullstops
    return text

def clean_arabic_text_semantic(text):
    # For semantic search normalization
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    return text

def simple_tokenize(text):
    return text.split()

def preprocess(text):
    # For BM25 tokenization
    text = normalize_arabic(text)
    tokens = simple_tokenize(text)
    tokens = [t for t in tokens if t not in arabic_stopwords and len(t) > 1]
    return tokens

# ----------------------------
# STEP 1: Load & Chunk Text
# ----------------------------

def load_and_chunk_text(file_path, sentences_per_chunk=5, cleaned_output_path="cleaned_output.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Clean the text
    text = clean_arabic_text_gui(text)

    # Save cleaned text for reference
    with open(cleaned_output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    # Split text into sentences
    sentences = re.split(r'(?<=[.؟!])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    # Create overlapping chunks
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)

    return chunks, sentences

# ----------------------------
# STEP 2: Classical Search with BM25
# ----------------------------

def setup_classical_search(chunks):
    # Preprocess documents for BM25
    tokenized_corpus = [preprocess(doc) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def classical_search(query, bm25, chunks, top_k=5):
    # Preprocess the query
    tokenized_query = preprocess(query)
    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)
    # Get top documents
    top_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for doc_idx, score in top_docs:
        results.append((chunks[doc_idx], score))
    
    return results

# ----------------------------
# STEP 3: Semantic Search with FAISS
# ----------------------------

def generate_embeddings(text_chunks, model):
    print("Generating embeddings...")
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    return embeddings

def build_faiss_index(embeddings):
    # Build FAISS index for cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def semantic_search(query, model, index, chunks, top_k=5):
    # Encode query
    query_embedding = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search index
    D, I = index.search(query_embedding, top_k)
    
    # Format results
    results = []
    for idx, i in enumerate(I[0]):
        if idx < len(D[0]):  # Ensure we don't go out of bounds
            results.append((chunks[i], D[0][idx]))
    
    return results

# ----------------------------
# STEP 4: Hybrid Retrieval
# ----------------------------

def hybrid_retrieve(query, model, faiss_index, documents, top_k_semantic=20, top_k_final=10):
    query_embedding = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)
    D, I = faiss_index.search(query_embedding, top_k_semantic)

    candidate_chunks = [documents[i] for i in I[0]]
    tokenized_query = preprocess(query)
    candidate_tokens = [preprocess(doc) for doc in candidate_chunks]

    bm25 = BM25Okapi(candidate_tokens)
    scores = bm25.get_scores(tokenized_query)
    top_reranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k_final]

    top_chunks = [candidate_chunks[idx] for idx, _ in top_reranked]
    return top_chunks

# ----------------------------
# STEP 5: Save/Load Functions
# ----------------------------

def save_embeddings_and_index(embeddings, index, chunks, bm25_model, 
                             index_file='index.faiss', 
                             embed_file='embeddings.npz', 
                             chunk_file='chunks.pkl',
                             bm25_file='bm25.pkl'):
    # Save FAISS index
    faiss.write_index(index, index_file)
    
    # Save embeddings
    np.savez(embed_file, embeddings=embeddings)
    
    # Save chunks
    with open(chunk_file, 'wb') as f:
        pickle.dump(chunks, f)
    
    # Save BM25 model
    with open(bm25_file, 'wb') as f:
        pickle.dump(bm25_model, f)

def load_embeddings_and_index(index_file='index.faiss', 
                             embed_file='embeddings.npz', 
                             chunk_file='chunks.pkl',
                             bm25_file='bm25.pkl'):
    # Load FAISS index
    index = faiss.read_index(index_file)
    
    # Load embeddings
    embeddings = np.load(embed_file)['embeddings']
    
    # Load chunks
    with open(chunk_file, 'rb') as f:
        chunks = pickle.load(f)
    
    # Load BM25 model
    with open(bm25_file, 'rb') as f:
        bm25_model = pickle.load(f)
    
    return embeddings, index, chunks, bm25_model

## Prepare Embeddings
file_path = "dB.txt"
index_file = 'index.faiss'
embed_file = 'embeddings.npz'
chunk_file = 'chunks.pkl'
bm25_file = 'bm25.pkl'

# Load or create embeddings and indexes
if os.path.exists(index_file) and os.path.exists(embed_file) and os.path.exists(chunk_file) and os.path.exists(bm25_file):
    print("Loading saved embeddings and index...")
    embeddings, faiss_index, chunks, bm25_model = load_embeddings_and_index(
        index_file, embed_file, chunk_file, bm25_file
    )
    print("Loaded saved data successfully!")
else:
    print("Processing text and generating new indexes...")
    # Load and chunk text
    chunks, sentences = load_and_chunk_text(file_path)
    
    # Load embedding model
    emb_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    print("Embedding model loaded.")
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks, emb_model)
    print("Embeddings generated.")
    
    # Build FAISS index
    faiss_index = build_faiss_index(embeddings)
    print("FAISS index built.")
    
    # Setup BM25 model
    bm25_model = setup_classical_search(chunks)
    print("BM25 model created.")
    
    # Save everything for future use
    save_embeddings_and_index(embeddings, faiss_index, chunks, bm25_model,
                                index_file, embed_file, chunk_file, bm25_file)
    print("All data saved for future use.")

# The embedding model for query encoding
emb_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
print("Embedding model loaded for queries.")

if __name__ == "__main__":
    Process(target=poll_new_messages).start()
    app.run(port=5000)