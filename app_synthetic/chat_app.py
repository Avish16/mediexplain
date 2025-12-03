import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# ----------------------------------------------------
# INIT
# ----------------------------------------------------
st.set_page_config(page_title="MediExplain Chatbot", layout="wide")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ----------------------------------------------------
# SESSION STATE
# ----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat memory

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""  # extracted medical report text


# ----------------------------------------------------
# TITLE + MODE SWITCH
# ----------------------------------------------------
st.title("🩺 MediExplain – Your Medical Report Companion")

mode = st.radio(
    "Choose AI Explanation Mode:",
    ["Patient Mode (Simple & Friendly)", "Caregiver Mode (Technical & Clinical)"]
)


# ----------------------------------------------------
# PDF UPLOAD & EXTRACTION
# ----------------------------------------------------
uploaded_pdf = st.file_uploader("Upload your medical report (PDF)", type=["pdf"])

if uploaded_pdf is not None:
    reader = PdfReader(uploaded_pdf)
    extracted = ""

    for page in reader.pages:
        extracted += page.extract_text() + "\n"

    st.session_state.pdf_text = extracted
    st.success("PDF uploaded and processed successfully!")

    with st.expander("📄 View extracted report text"):
        st.write(st.session_state.pdf_text)


# ----------------------------------------------------
# CHAT HISTORY UI
# ----------------------------------------------------
st.write("Conversation")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])


# ----------------------------------------------------
# RESPONSE GENERATION FUNCTION
# ----------------------------------------------------
def generate_response(user_input, mode):
    # Persona based on selected mode
    if "Patient Mode" in mode:
        persona = "Use simple, friendly, non-technical language. Avoid medical jargon."
    else:
        persona = "Use clinical accuracy, medical terminology, and deeper reasoning."

    # Build conversation for the model
    messages = [
        {
            "role": "system",
            "content": f"""
You are MediExplain, an AI assistant that explains medical information clearly.

Persona rules:
{persona}

Here is the patient's uploaded medical report:
-------------------
{st.session_state.pdf_text}
-------------------
"""
        }
    ]

    # Add memory from previous turns
    for m in st.session_state.messages:
        messages.append({"role": m["role"], "content": m["content"]})

    # Add the new question
    messages.append({"role": "user", "content": user_input})

    # Call the model
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=messages
    )

    return response.choices[0].message.content


# ----------------------------------------------------
# CHAT INPUT (STREAMLIT CHAT BOX)
# ----------------------------------------------------
user_input = st.chat_input("Ask a question about your medical report...")

if user_input:
    # Save user's message to chat memory
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate AI response
    answer = generate_response(user_input, mode)

    # Save AI response
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Display message
    st.chat_message("assistant").markdown(answer)


# ----------------------------------------------------
# RESET BUTTON
# ----------------------------------------------------
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()
