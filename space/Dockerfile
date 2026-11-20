# Callie public demo: the customer phone line, in one Hugging Face Docker Space.
# Built from a clean copy of the repo by scripts/deploy_space.sh (this file becomes /Dockerfile).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    GRADIO_ANALYTICS_ENABLED=False

# The demo's own Postgres (fresh demo data on every start: no external database needed).
RUN apt-get update && apt-get install -y --no-install-recommends postgresql \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-callie.txt ./
RUN pip install -r requirements-callie.txt

# Models baked into the image: TalkTable's embeddings, Whisper base.en (the fallback when
# Groq's speech-to-text is busy), and the Piper voice.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
 && python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')" \
 && mkdir -p /app/data/voices \
 && python -m piper.download_voices en_US-lessac-medium --download-dir /app/data/voices
ENV HF_HUB_OFFLINE=1

COPY . .

# Hugging Face runs Spaces as uid 1000. It may write only its own files: Postgres in /tmp,
# the staff/service users file, the audit log and the answer cache in /app/data.
RUN useradd --create-home --uid 1000 user \
    && mkdir -p /app/data/audit /app/.gradio \
    && chown -R user /app/data /app/.gradio /opt/hf-cache
USER user

# The demo's settings (secrets come from the Space's settings page: GROQ_API_KEY, and
# for the phone line TWILIO_AUTH_TOKEN / TWILIO_ACCOUNT_SID). The models are measured in
# docs/callie_plan.md (H2).
ENV AGENT_PROVIDER=groq \
    AGENT_MODEL=openai/gpt-oss-120b \
    AGENT_REASONING_EFFORT=low \
    AGENT_FALLBACK_MODELS=openai/gpt-oss-20b@low,qwen/qwen3.8-27b@none \
    AGENT_MAX_RETRIES=1 \
    CALLIE_TALKTABLE_LLM=default \
    LLM_PROVIDER=groq \
    LLM_MODEL=qwen/qwen3.8-27b \
    PHONE_STT=groq \
    PHONE_STT_MODEL=base.en \
    PHONE_MAX_CALLS=3 \
    PHONE_MAX_MINUTES=5 \
    PHONE_CALLS_PER_NUMBER_HOUR=4 \
    CACHE_ENABLED=true

EXPOSE 7860
CMD ["bash", "space/start.sh"]
