import os
import io
import tempfile
import base64
import random
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from gtts import gTTS
from crewai import Agent, Task, Crew, Process
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="English Fluency Coach", page_icon="🎓")
st.title("🎓 English Fluency Coach")
st.caption("Have a real conversation — I'll help you sound more natural!")

groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("API Key missing! Please set GROQ_API_KEY in Streamlit Secrets.")
    st.stop()
os.environ["GROQ_API_KEY"] = groq_key
groq_client = Groq(api_key=groq_key)

MAX_TURNS = 5

# ── Topics ────────────────────────────────────────────────────────────────────
TOPICS = [
    {"name": "At a coffee shop",          "opening": "I just tried that new café on the corner. Have you been there yet?"},
    {"name": "Weekend plans",             "opening": "So what are you up to this weekend? Got anything fun planned?"},
    {"name": "Talking about the weather", "opening": "Can you believe how hot it's been lately? Is it like this where you live?"},
    {"name": "Recommending a restaurant", "opening": "I had the most amazing dinner last night. Do you have a favourite restaurant around here?"},
    {"name": "Talking about a movie",     "opening": "I watched a really good movie last night. Have you seen anything good recently?"},
    {"name": "Monday morning small talk", "opening": "Hey, how was your weekend? Do anything interesting?"},
    {"name": "Talking about a hobby",     "opening": "I've been trying to get into running lately. Do you have any hobbies you're really into?"},
    {"name": "Planning a trip",           "opening": "I'm thinking about taking a trip somewhere next month. Have you travelled anywhere nice lately?"},
    {"name": "Talking about food",        "opening": "I've been trying to cook more at home. Do you enjoy cooking?"},
    {"name": "Catching up with a friend", "opening": "It feels like we haven't talked in ages! What have you been up to lately?"},
    {"name": "Talking about work",        "opening": "Work has been so busy for me lately. How about you — are things busy at your end?"},
    {"name": "Talking about a TV show",   "opening": "I just finished watching a really good series. Are you watching anything good right now?"},
    {"name": "Shopping",                  "opening": "I went to the mall yesterday and it was packed! Do you enjoy shopping?"},
    {"name": "Health and exercise",       "opening": "I've been trying to go to the gym more regularly. Do you exercise much?"},
    {"name": "Talking about pets",        "opening": "My neighbour just got a puppy and it's so cute! Do you have any pets?"},
]

# ── Agent ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    return Agent(
        role="English Fluency Coach",
        goal="Have a natural everyday conversation while coaching spoken English fluency.",
        backstory="""You are a friendly English fluency coach having a real casual conversation
        with a non-native English speaker.

        For EVERY student message, respond in exactly this format — two lines, nothing else:

        Comment: <fluency feedback>
        Question: <follow-up question>

        Rules for Comment:
        - If UNNATURAL: say "A more natural way to say this is: ..." then explain why in one sentence.
        - If already NATURAL: short warm response like "That sounds great!" — no rewrites.

        Rules for Question:
        - One short natural follow-up question that fits the conversation.
        - Sound like a real person chatting, not a teacher.
        - Never repeat a question already asked.""",
        llm="groq/llama-3.3-70b-versatile",
        verbose=False,
    )

# ── Helpers ───────────────────────────────────────────────────────────────────
def transcribe(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()
        with open(f.name, "rb") as af:
            result = groq_client.audio.transcriptions.create(
                model="whisper-large-v3", file=af, language="en",
            )
    return result.text.strip()

def get_response(student_text: str, history: list, topic_name: str) -> tuple[str, str]:
    history_str = ""
    for msg in history[-8:]:
        role = "Student" if msg["role"] == "user" else "Coach"
        history_str += f"{role}: {msg['content']}\n"
    task = Task(
        description=(
            f"Topic: {topic_name}\n"
            f"Conversation so far:\n{history_str}\n"
            f"Student just said: \"{student_text}\"\n\n"
            f"Respond using exactly:\nComment: <feedback>\nQuestion: <follow-up question>"
        ),
        expected_output="Comment: <feedback>\nQuestion: <follow-up question>",
        agent=get_agent(),
    )
    crew = Crew(agents=[get_agent()], tasks=[task], process=Process.sequential)
    raw = crew.kickoff().raw
    comment, question = "", ""
    for line in raw.strip().splitlines():
        if line.lower().startswith("comment:"):
            comment = line[len("comment:"):].strip()
        elif line.lower().startswith("question:"):
            question = line[len("question:"):].strip()
    return comment or raw.strip(), question

def make_audio_b64(text: str) -> str:
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def autoplay_audio(b64: str):
    st.markdown(
        f'<audio autoplay style="display:none">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
        unsafe_allow_html=True,
    )

# ── Review component ──────────────────────────────────────────────────────────
def render_review_component(speeches: list[str], comments: list[str], comment_audios: list[str]):
    """
    Self-contained HTML component that handles the full review flow.
    Each card shows the user's speech + comment text.
    Clicking 'Hear comment' plays the audio within the same gesture — reliable across all browsers.
    Clicking 'Next' advances the card.
    """
    # Serialize data safely for JS
    import json
    speeches_js  = json.dumps(speeches)
    comments_js  = json.dumps(comments)
    audios_js    = json.dumps(comment_audios)
    total        = len(speeches)

    html = f"""
    <style>
      * {{ box-sizing: border-box; font-family: sans-serif; }}
      body {{ margin: 0; padding: 12px; background: transparent; }}
      .card {{ background: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 12px; border: 1px solid #dee2e6; }}
      .label {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; color: #6c757d; margin-bottom: 4px; }}
      .speech {{ font-size: 15px; color: #212529; margin-bottom: 14px; line-height: 1.5; }}
      .comment {{ font-size: 15px; color: #0d6efd; line-height: 1.5; margin-bottom: 16px; }}
      .progress {{ font-size: 12px; color: #6c757d; margin-bottom: 14px; }}
      .btn {{ display: inline-block; padding: 9px 20px; border-radius: 7px; border: none;
              font-size: 14px; font-weight: 500; cursor: pointer; margin-right: 8px; }}
      .btn-primary {{ background: #0d6efd; color: white; }}
      .btn-primary:hover {{ background: #0b5ed7; }}
      .btn-secondary {{ background: #e9ecef; color: #495057; }}
      .btn-secondary:hover {{ background: #dee2e6; }}
      .done {{ background: #d1e7dd; border-radius: 10px; padding: 16px 20px; font-size: 15px; color: #0a3622; }}
    </style>

    <div id="app"></div>

    <script>
    const speeches = {speeches_js};
    const comments = {comments_js};
    const audios   = {audios_js};
    const total    = {total};
    let current    = 0;
    let playing    = false;

    function render() {{
      const app = document.getElementById('app');
      if (current >= total) {{
        app.innerHTML = `
          <div class="done">
            ✅ Review complete! Great work today.<br>
            <small>Click <strong>🔄 New topic</strong> below to start a new conversation.</small>
          </div>`;
        return;
      }}

      app.innerHTML = `
        <div class="progress">Comment ${{current + 1}} of ${{total}}</div>
        <div class="card">
          <div class="label">You said</div>
          <div class="speech">${{speeches[current]}}</div>
          <div class="label">Fluency feedback</div>
          <div class="comment">${{comments[current]}}</div>
          <button class="btn btn-primary" onclick="playComment()" id="play-btn">🔊 Hear comment</button>
          <button class="btn btn-secondary" onclick="nextCard()"
            id="next-btn">${{current + 1 < total ? 'Next comment ▶' : 'Finish review ✓'}}</button>
        </div>`;
    }}

    function playComment() {{
      if (playing) return;
      playing = true;
      document.getElementById('play-btn').textContent = '🔊 Playing...';
      const audio = new Audio('data:audio/mp3;base64,' + audios[current]);
      audio.onended = () => {{
        playing = false;
        const btn = document.getElementById('play-btn');
        if (btn) btn.textContent = '🔊 Hear comment';
      }};
      audio.play();
    }}

    function nextCard() {{
      current++;
      render();
    }}

    // Auto-play first comment on load
    render();
    setTimeout(() => {{ playComment(); }}, 400);
    </script>
    """
    components.html(html, height=320, scrolling=False)

# ── Session state ─────────────────────────────────────────────────────────────
if "topic"          not in st.session_state: st.session_state.topic          = random.choice(TOPICS)
if "messages"       not in st.session_state: st.session_state.messages       = []
if "speeches"       not in st.session_state: st.session_state.speeches       = []
if "comments"       not in st.session_state: st.session_state.comments       = []
if "turn_count"     not in st.session_state: st.session_state.turn_count     = 0
if "audio_enabled"  not in st.session_state: st.session_state.audio_enabled  = True
if "pending_audio"  not in st.session_state: st.session_state.pending_audio  = None
if "started"        not in st.session_state: st.session_state.started        = False
if "finished"       not in st.session_state: st.session_state.finished       = False
if "comment_audios" not in st.session_state: st.session_state.comment_audios = []

topic = st.session_state.topic

# ── Audio toggle ──────────────────────────────────────────────────────────────
col_tog, col_topic, _ = st.columns([1, 2, 3])
with col_tog:
    label = "🔊 Audio ON" if st.session_state.audio_enabled else "🔇 Audio OFF"
    if st.button(label, use_container_width=True):
        st.session_state.audio_enabled = not st.session_state.audio_enabled
        st.rerun()
with col_topic:
    st.markdown(f"**Topic:** {topic['name']}")

st.divider()

# ── Opening message ───────────────────────────────────────────────────────────
if not st.session_state.started:
    opening = topic["opening"]
    st.session_state.messages.append({"role": "assistant", "content": opening})
    st.session_state.pending_audio = make_audio_b64(opening)
    st.session_state.started = True

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Play single pending audio (questions only) ────────────────────────────────
if st.session_state.audio_enabled and st.session_state.pending_audio:
    autoplay_audio(st.session_state.pending_audio)
    st.session_state.pending_audio = None

# ── FINISHED: render review component ────────────────────────────────────────
if st.session_state.finished:
    st.markdown("---")
    st.markdown("### 📝 Fluency Review")
    st.caption("Press **🔊 Hear comment** to listen. Press **Next comment** to advance.")

    render_review_component(
        st.session_state.speeches,
        st.session_state.comments,
        st.session_state.comment_audios,
    )

    if st.button("🔄 New topic", type="primary"):
        for key in ["topic", "messages", "speeches", "comments", "turn_count",
                    "pending_audio", "started", "finished", "comment_audios"]:
            del st.session_state[key]
        st.rerun()
    st.stop()

# ── Progress ──────────────────────────────────────────────────────────────────
st.caption(f"Turn {min(st.session_state.turn_count + 1, MAX_TURNS)} of {MAX_TURNS}")
st.progress(st.session_state.turn_count / MAX_TURNS)
st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("**🎤 Tap to start · tap again to stop**")
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e85d04",
        neutral_color="#6c757d",
        icon_size="2x",
        pause_threshold=3.0,
        key=f"recorder_{st.session_state.turn_count}",
    )
with col2:
    st.markdown("**⌨️ Or type your answer:**")
    text_input = st.chat_input("Type here...")

# ── Process answer ────────────────────────────────────────────────────────────
def handle_answer(user_text: str):
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.turn_count += 1
    is_last = st.session_state.turn_count >= MAX_TURNS

    with st.spinner("Thinking..."):
        comment, question = get_response(user_text, st.session_state.messages, topic["name"])

    # Save speech + comment + pre-generate audio for review
    st.session_state.speeches.append(user_text)
    st.session_state.comments.append(comment)
    st.session_state.comment_audios.append(make_audio_b64(comment))

    # Show comment as text in chat
    st.session_state.messages.append({"role": "assistant", "content": f"💬 {comment}"})

    if is_last:
        closing = "Great conversation! Here's your fluency review."
        st.session_state.messages.append({"role": "assistant", "content": closing})
        st.session_state.pending_audio = make_audio_b64(closing)
        st.session_state.finished = True
    else:
        if question:
            st.session_state.messages.append({"role": "assistant", "content": question})
            st.session_state.pending_audio = make_audio_b64(question)

    st.rerun()

if audio_bytes and len(audio_bytes) > 1000:
    with st.spinner("Transcribing..."):
        try:
            user_text = transcribe(audio_bytes)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            user_text = None
    if user_text:
        handle_answer(user_text)
elif text_input:
    handle_answer(text_input)
