import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

# Load API Key
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# Streamlit Page Config
st.set_page_config(
    page_title="PDF Q&A Generator",
    page_icon="⚡",
    layout="wide"
)

# Custom Dark Theme & UI Enhancements
st.markdown("""
<style>
    /* Dark Theme Core Styles */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Header Gradient & Typography */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff7b72, #d2a8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Large Header Icon Box */
    .icon-container {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }

    /* Metric & File Cards */
    .dark-card {
        background-color: #161b22;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid #30363d;
        border-left: 5px solid #d2a8ff;
        margin-bottom: 1.5rem;
    }

    /* Q&A Output Cards */
    .qa-card {
        background-color: #161b22;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        border: 1px solid #30363d;
    }

    /* Highlight Pill Tags */
    .section-pill {
        display: inline-block;
        background-color: #21262d;
        color: #58a6ff;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #30363d;
        margin-bottom: 12px;
    }

    /* Dark Sidebar Customization */
    div[data-testid="stSidebar"] {
        background-color: #010409;
        border-right: 1px solid #21262d;
    }
</style>
""", unsafe_allow_html=True)

# Main UI Header with Large Icon
st.markdown('<div class="icon-container">⚡📄</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">AI PDF Q&A Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Transform static PDF documents into high-yield study guides and structured question banks.</div>', unsafe_allow_html=True)

# Sidebar - Workspace Config & Performance Controls
with st.sidebar:
    st.title("⚙️ Workspace Config")
    st.markdown("---")
    
    # API Key Configuration
    if not api_key:
        api_key = st.text_input("Enter Mistral API Key:", type="password")
        st.caption("🔑 Required to initialize LLM capabilities.")
    else:
        st.success("✔ API Key loaded from environment")

    st.markdown("---")
    st.subheader("🚀 Performance Tuning")
    
    # Model Selection
    model_choice = st.selectbox(
        "Select Model Quality",
        ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
        help="Higher quality models provide better analytical precision but take slightly longer."
    )
    
    # Speed vs Accuracy (Temperature Control)
    creativity = st.slider(
        "Response Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Lower values focus strictly on facts; higher values generate more varied questions."
    )

    st.markdown("---")
    st.subheader("🎯 Focus & Extraction")
    
    # Highlight Priority Questions
    extract_priority = st.checkbox(
        "Highlight High-Priority Questions",
        value=True,
        help="Instructs the AI to flag critical, exam-worthy topics with high-priority tags."
    )

# Main Workspace - File Input
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file:
    if not api_key:
        st.error("⚠️ Please provide a valid Mistral API Key in the sidebar workspace controls to process the document.")
    else:
        # File Summary Card
        st.markdown(
            f"""
            <div class="dark-card">
                <span style="color: #8b949e; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.5px;">LOADED FILE</span><br>
                <span style="font-size: 1.25rem; font-weight: 700; color: #f0f6fc;">📑 {uploaded_file.name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("🚀 Generate Q&A Content", type="primary", use_container_width=True):
            try:
                with st.spinner("Analyzing document structure and preparing chunk blocks..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_pdf_path = tmp_file.name

                    loader = PyPDFLoader(tmp_pdf_path)
                    docs = loader.load()
                    os.remove(tmp_pdf_path)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=2500,
                        chunk_overlap=200
                    )
                    chunks = splitter.split_documents(docs)

                st.success(f"Processing **{len(chunks)}** logical text block(s) with **{model_choice}**...")

                llm = ChatMistralAI(
                    model=model_choice,
                    api_key=api_key,
                    temperature=creativity
                )

                all_qa_blocks = []
                progress_bar = st.progress(0)

                # Custom Prompt Construction based on Config
                priority_instruction = ""
                if extract_priority:
                    priority_instruction = "- Mark the most critical/high-yield questions with a '🔥 **[IMPORTANT]**' tag at the start of the question line."

                for idx, chunk in enumerate(chunks):
                    prompt = f"""
You are an expert tutor and subject specialist. Analyze the following text extracted from a PDF and generate all possible key Questions and Answers that thoroughly cover the content.

Formatting instructions:
- **Q:** Clear and direct question
- **A:** Complete and informative answer
{priority_instruction}

Text context:
{chunk.page_content}
"""
                    response = llm.invoke([HumanMessage(content=prompt)])
                    all_qa_blocks.append({"section": idx + 1, "content": response.content})
                    progress_bar.progress((idx + 1) / len(chunks))

                st.session_state["qa_result"] = all_qa_blocks
                st.session_state["file_name"] = uploaded_file.name

            except Exception as e:
                st.error(f"Processing Error: {str(e)}")

# Display Generated Output
if "qa_result" in st.session_state:
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📌 Generated Q&A Dataset")
    with col2:
        raw_text = "\n\n---\n\n".join([f"### Section {item['section']}\n\n{item['content']}" for item in st.session_state["qa_result"]])
        st.download_button(
            label="📥 Download Output (.txt)",
            data=raw_text,
            file_name=f"QnA_{st.session_state['file_name']}.txt",
            mime="text/plain",
            use_container_width=True
        )

    # Render Q&A text inside visual dark cards
    for item in st.session_state["qa_result"]:
        st.markdown(
            f"""
            <div class="qa-card">
                <span class="section-pill">Section {item['section']}</span>
                <div>{item['content']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )