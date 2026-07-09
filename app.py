import os
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from pydantic import SecretStr

# ---------------------------------------------------------------------------
# Environment & Configuration
# ---------------------------------------------------------------------------
load_dotenv()

IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
PROJECT_ID        = os.getenv("PROJECT_ID")
IBM_WATSONX_URL   = os.getenv("IBM_WATSONX_URL", "https://au-syd.ml.cloud.ibm.com")

# ---------------------------------------------------------------------------
# RAG Pipeline Setup
# ---------------------------------------------------------------------------

def build_vectorstore(filepath: str = "university_data.txt") -> FAISS:
    """Load university_data.txt, split it, and embed into a FAISS vectorstore."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.create_documents([raw_text])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def build_chain(vectorstore: FAISS) -> ConversationalRetrievalChain:
    """Wire up the LLM + FAISS retriever into a ConversationalRetrievalChain."""
    llm = WatsonxLLM(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url=SecretStr(IBM_WATSONX_URL),
        apikey=SecretStr(IBM_CLOUD_API_KEY) if IBM_CLOUD_API_KEY else None,
        project_id=PROJECT_ID,
        verify=False,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 150,
            "min_new_tokens": 1,
            "temperature": 0,
            "stop_sequences": ["\n\n", "Question:", "Q:"],
        },
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a concise admission assistant for New Era Engineering College. "
            "Answer using ONLY the provided context. "
            "Reply in 1-3 sentences. "
            "Output the answer text only — no labels, no follow-up questions, no extra commentary. "
            "If the answer is not in the context, say: I'm sorry, I don't have that information.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "Context:\n{context}\n\n"
            "{question}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        ),
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=False,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )
    return chain


# Build once at startup
print("⏳  Building FAISS vectorstore …")
vectorstore = build_vectorstore()
print("⏳  Initialising Granite LLM chain …")
qa_chain = build_chain(vectorstore)
print("✅  RAG Agent ready.")

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>New Era Engineering College — Admission Assistant</title>
  <!-- Bootstrap 5 -->
  <link
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
    rel="stylesheet"
  />
  <style>
    /* ── Brand tokens ── */
    :root {
      --brand-primary:   #003087;   /* deep university blue  */
      --brand-secondary: #0057b8;   /* mid blue              */
      --brand-accent:    #f5a800;   /* gold accent           */
      --brand-light:     #e8f0fb;   /* pale blue surface     */
      --brand-white:     #ffffff;
      --radius:          0.6rem;
    }

    body {
      background: var(--brand-light);
      font-family: "Segoe UI", system-ui, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* ── Top nav ── */
    .navbar-brand span.highlight { color: var(--brand-accent); font-weight: 700; }

    /* ── Card container ── */
    .chat-card {
      border: none;
      border-radius: var(--radius);
      box-shadow: 0 4px 24px rgba(0,0,0,.10);
      display: flex;
      flex-direction: column;
      height: 72vh;
    }

    .chat-card .card-header {
      background: var(--brand-primary);
      color: var(--brand-white);
      border-radius: var(--radius) var(--radius) 0 0;
      padding: .9rem 1.2rem;
    }

    /* ── Message window ── */
    #chat-window {
      flex: 1;
      overflow-y: auto;
      padding: 1.2rem;
      background: var(--brand-white);
      display: flex;
      flex-direction: column;
      gap: .75rem;
    }

    /* ── Bubbles ── */
    .bubble-wrapper { display: flex; }
    .bubble-wrapper.user  { justify-content: flex-end; }
    .bubble-wrapper.agent { justify-content: flex-start; }

    .bubble {
      max-width: 72%;
      padding: .55rem .9rem;
      border-radius: 1.1rem;
      font-size: .93rem;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .bubble-wrapper.user  .bubble {
      background: var(--brand-secondary);
      color: var(--brand-white);
      border-bottom-right-radius: .25rem;
    }
    .bubble-wrapper.agent .bubble {
      background: var(--brand-light);
      color: #1a1a2e;
      border-bottom-left-radius: .25rem;
      border: 1px solid #c7d8f5;
    }

    .avatar {
      width: 32px; height: 32px;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-weight: 700; font-size: .78rem;
      flex-shrink: 0;
    }
    .bubble-wrapper.user  .avatar { background: var(--brand-accent); color: #000; margin-left: .5rem; }
    .bubble-wrapper.agent .avatar { background: var(--brand-primary); color: #fff; margin-right: .5rem; }

    /* ── Typing indicator ── */
    .typing-dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--brand-secondary);
      display: inline-block;
      animation: blink 1.2s infinite;
    }
    .typing-dot:nth-child(2) { animation-delay: .2s; }
    .typing-dot:nth-child(3) { animation-delay: .4s; }
    @keyframes blink {
      0%, 80%, 100% { opacity: .2; transform: scale(.9); }
      40%           { opacity: 1;  transform: scale(1.1); }
    }

    /* ── Input bar ── */
    .chat-card .card-footer {
      background: #f0f4fb;
      border-top: 1px solid #c7d8f5;
      border-radius: 0 0 var(--radius) var(--radius);
      padding: .75rem 1rem;
    }
    #user-input {
      border: 1.5px solid #b3c8ed;
      border-radius: 2rem;
      padding: .5rem 1rem;
      font-size: .93rem;
      resize: none;
      transition: border-color .2s;
    }
    #user-input:focus { border-color: var(--brand-secondary); outline: none; box-shadow: none; }

    #send-btn {
      background: var(--brand-primary);
      color: var(--brand-white);
      border: none;
      border-radius: 2rem;
      padding: .5rem 1.4rem;
      font-weight: 600;
      transition: background .2s;
    }
    #send-btn:hover:not(:disabled) { background: var(--brand-secondary); }
    #send-btn:disabled { opacity: .6; cursor: not-allowed; }

    /* ── Footer ── */
    footer { font-size: .78rem; color: #6c757d; }
  </style>
</head>
<body>

<!-- ── Navbar ── -->
<nav class="navbar navbar-dark" style="background:var(--brand-primary);">
  <div class="container">
    <span class="navbar-brand mb-0 h1">
      🎓 New Era Engineering College &nbsp;|&nbsp;
      <span class="highlight">Admission Assistant</span>
    </span>
  </div>
</nav>

<!-- ── Main ── -->
<main class="container my-4 flex-grow-1">
  <div class="row justify-content-center">
    <div class="col-12 col-lg-8 col-xl-7">

      <!-- Info badge -->
      <div class="alert alert-info py-2 px-3 mb-3" role="alert" style="font-size:.88rem;">
        <strong>Powered by IBM Watsonx Granite</strong> — Ask me anything about
        admissions, fees, eligibility, or hostel facilities.
      </div>

      <!-- Chat card -->
      <div class="card chat-card">
        <div class="card-header d-flex align-items-center gap-2">
          <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round"
              d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6
                 a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-3 3v-3z"/>
          </svg>
          <span class="fw-semibold">Chat with the Admission Bot</span>
        </div>

        <div id="chat-window">
          <!-- Greeting -->
          <div class="bubble-wrapper agent">
            <div class="avatar">AI</div>
            <div class="bubble">
              Hello! 👋 I'm the New Era Engineering College Admission Assistant.
              I can help you with eligibility criteria, fees, deadlines, and hostel
              details. How can I assist you today?
            </div>
          </div>
        </div>

        <div class="card-footer">
          <div class="d-flex gap-2 align-items-end">
            <textarea
              id="user-input"
              class="form-control"
              rows="1"
              placeholder="Type your question here…"
              style="flex:1;"
            ></textarea>
            <button id="send-btn" onclick="sendMessage()">Send</button>
          </div>
        </div>
      </div>

    </div>
  </div>
</main>

<footer class="text-center py-3">
  New Era Engineering College &copy; 2026 &mdash; RAG Agent powered by
  <strong>IBM Watsonx</strong> &amp; LangChain
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
  const chatWindow = document.getElementById("chat-window");
  const inputEl    = document.getElementById("user-input");
  const sendBtn    = document.getElementById("send-btn");

  /* Auto-grow textarea */
  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
  });

  /* Send on Enter (Shift+Enter = newline) */
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  function appendBubble(role, text) {
    const wrapper = document.createElement("div");
    wrapper.className = `bubble-wrapper ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "You" : "AI";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    if (role === "user") {
      wrapper.appendChild(bubble);
      wrapper.appendChild(avatar);
    } else {
      wrapper.appendChild(avatar);
      wrapper.appendChild(bubble);
    }

    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return bubble;
  }

  function showTyping() {
    const wrapper = document.createElement("div");
    wrapper.className = "bubble-wrapper agent";
    wrapper.id = "typing-indicator";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "AI";

    const bubble = document.createElement("div");
    bubble.className = "bubble d-flex gap-1 align-items-center";
    bubble.innerHTML =
      '<span class="typing-dot"></span>' +
      '<span class="typing-dot"></span>' +
      '<span class="typing-dot"></span>';

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function removeTyping() {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
  }

  async function sendMessage() {
    const message = inputEl.value.trim();
    if (!message) return;

    /* Render user bubble & clear input */
    appendBubble("user", message);
    inputEl.value = "";
    inputEl.style.height = "auto";
    sendBtn.disabled = true;
    showTyping();

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      removeTyping();

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        appendBubble("agent",
          "⚠️ Error " + response.status + ": " +
          (err.error || "Something went wrong. Please try again.")
        );
        return;
      }

      const data = await response.json();
      appendBubble("agent", data.answer || "I'm sorry, I couldn't generate a response.");

    } catch (networkError) {
      removeTyping();
      appendBubble("agent",
        "⚠️ Network error: Unable to reach the server. " +
        "Please check your connection and try again."
      );
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or not data.get("message", "").strip():
        return jsonify({"error": "No message provided."}), 400

    user_message = data["message"].strip()

    try:
        result = qa_chain.invoke({"question": user_message})
        answer = result.get("answer", "I'm sorry, I couldn't find an answer.")
        # Strip anything the model generated after its first answer
        # (e.g. self-generated follow-up "Question: ... Answer: ..." blocks)
        answer = re.split(r'\n\s*(?:Question|Q)\s*:', answer, maxsplit=1)[0].strip()
        return jsonify({"answer": answer})
    except Exception as exc:
        app.logger.error("Chain error: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
