"""
MediExplain Home / Landing Page
"""
import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from styles import inject_global_css

inject_global_css()

# ── HERO ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 60px 20px 40px; animation: fadeIn 0.6s ease;">

  <div style="display:inline-block; background:rgba(0,212,255,0.08); border:1px solid rgba(0,212,255,0.2);
              border-radius:100px; padding:6px 18px; font-size:12px; color:#00D4FF;
              font-family:'Inter',sans-serif; letter-spacing:2px; font-weight:600;
              text-transform:uppercase; margin-bottom:24px;">
    Clinical AI Platform
  </div>

  <h1 style="font-family:'Space Grotesk',sans-serif; font-size:clamp(40px,6vw,72px);
             font-weight:700; margin:0 0 16px;
             background:linear-gradient(135deg,#F1F5F9 0%,#00D4FF 50%,#0066FF 100%);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;
             background-clip:text; line-height:1.1;">
    MediExplain
  </h1>

  <p style="font-family:'Inter',sans-serif; font-size:clamp(16px,2.5vw,22px);
            color:#94A3B8; margin:0 auto 40px; max-width:600px; line-height:1.6; font-weight:300;">
    AI-Powered Medical Intelligence Platform
  </p>

  <div style="display:flex; justify-content:center; gap:32px; flex-wrap:wrap;
              font-family:'Inter',sans-serif; font-size:14px; color:#64748B; margin-bottom:52px;">
    <span style="display:flex;align-items:center;gap:6px;">
      <span style="color:#00D4FF;">●</span> 19 Specialized AI Bots
    </span>
    <span style="display:flex;align-items:center;gap:6px;">
      <span style="color:#00D4FF;">●</span> RAG-Enhanced Retrieval
    </span>
    <span style="display:flex;align-items:center;gap:6px;">
      <span style="color:#00D4FF;">●</span> Clinical-Grade PDF Output
    </span>
    <span style="display:flex;align-items:center;gap:6px;">
      <span style="color:#00D4FF;">●</span> Real-Time Safety Labeling
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── FEATURE CARDS ───────────────────────────────────────────────
st.markdown("""
<div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
            gap:20px; padding:0 4px 48px; animation:fadeIn 0.8s ease;">

  <div style="background:rgba(255,255,255,0.025); border:1px solid rgba(0,212,255,0.15);
              border-radius:16px; padding:28px; transition:all 0.3s ease;
              position:relative; overflow:hidden;">
    <div style="position:absolute;top:0;right:0;width:120px;height:120px;
                background:radial-gradient(circle,rgba(0,212,255,0.06),transparent 70%);"></div>

    <div style="width:48px;height:48px;border-radius:12px;
                background:linear-gradient(135deg,rgba(0,212,255,0.15),rgba(0,102,255,0.1));
                border:1px solid rgba(0,212,255,0.25);
                display:flex;align-items:center;justify-content:center;
                font-size:22px;margin-bottom:16px;">🧬</div>

    <div style="font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:600;
                color:#F1F5F9;margin-bottom:8px;">Synthetic Patient Generator</div>

    <p style="font-family:'Inter',sans-serif;font-size:14px;color:#64748B;
              line-height:1.6;margin:0 0 16px;">
      Generate complete, clinically realistic patient records end-to-end using
      a 19-bot AI pipeline — demographics, labs, vitals, clinical notes, prescriptions,
      billing, safety labels, and a full PDF report.
    </p>

    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:4px;">
      <span style="font-size:11px;color:#00D4FF;background:rgba(0,212,255,0.08);
                   padding:3px 10px;border-radius:100px;font-weight:500;">19 Bots</span>
      <span style="font-size:11px;color:#94A3B8;background:rgba(255,255,255,0.05);
                   padding:3px 10px;border-radius:100px;">PDF Export</span>
      <span style="font-size:11px;color:#94A3B8;background:rgba(255,255,255,0.05);
                   padding:3px 10px;border-radius:100px;">Safety Audit</span>
    </div>
  </div>

  <div style="background:rgba(255,255,255,0.025); border:1px solid rgba(0,212,255,0.15);
              border-radius:16px; padding:28px; transition:all 0.3s ease;
              position:relative; overflow:hidden;">
    <div style="position:absolute;top:0;right:0;width:120px;height:120px;
                background:radial-gradient(circle,rgba(0,102,255,0.07),transparent 70%);"></div>

    <div style="width:48px;height:48px;border-radius:12px;
                background:linear-gradient(135deg,rgba(0,102,255,0.15),rgba(0,212,255,0.1));
                border:1px solid rgba(0,102,255,0.25);
                display:flex;align-items:center;justify-content:center;
                font-size:22px;margin-bottom:16px;">💬</div>

    <div style="font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:600;
                color:#F1F5F9;margin-bottom:8px;">MediExplain Chatbot</div>

    <p style="font-family:'Inter',sans-serif;font-size:14px;color:#64748B;
              line-height:1.6;margin:0 0 16px;">
      Upload any medical report PDF and ask questions in plain language.
      Specialist bots handle labs, medications, care plans, diagnoses, and
      emotional support — powered by RAG retrieval and long-term memory.
    </p>

    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:4px;">
      <span style="font-size:11px;color:#6366F1;background:rgba(99,102,241,0.1);
                   padding:3px 10px;border-radius:100px;font-weight:500;">RAG-Enhanced</span>
      <span style="font-size:11px;color:#94A3B8;background:rgba(255,255,255,0.05);
                   padding:3px 10px;border-radius:100px;">7 Specialist Bots</span>
      <span style="font-size:11px;color:#94A3B8;background:rgba(255,255,255,0.05);
                   padding:3px 10px;border-radius:100px;">Memory</span>
    </div>
  </div>

  <div style="background:rgba(255,255,255,0.025); border:1px solid rgba(0,212,255,0.15);
              border-radius:16px; padding:28px; transition:all 0.3s ease;
              position:relative; overflow:hidden;">
    <div style="position:absolute;top:0;right:0;width:120px;height:120px;
                background:radial-gradient(circle,rgba(16,185,129,0.06),transparent 70%);"></div>

    <div style="width:48px;height:48px;border-radius:12px;
                background:linear-gradient(135deg,rgba(16,185,129,0.15),rgba(0,212,255,0.1));
                border:1px solid rgba(16,185,129,0.25);
                display:flex;align-items:center;justify-content:center;
                font-size:22px;margin-bottom:16px;">🔬</div>

    <div style="font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:600;
                color:#F1F5F9;margin-bottom:8px;">Validator Console</div>

    <p style="font-family:'Inter',sans-serif;font-size:14px;color:#64748B;
              line-height:1.6;margin:0 0 16px;">
      Developer-facing diagnostics console for inspecting retrieval quality,
      routing decisions, safety guardrails, bot outputs, and Q&A history.
      Full pipeline introspection in a single view.
    </p>

    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:4px;">
      <span style="font-size:11px;color:#10B981;background:rgba(16,185,129,0.1);
                   padding:3px 10px;border-radius:100px;font-weight:500;">Diagnostics</span>
      <span style="font-size:11px;color:#94A3B8;background:rgba(255,255,255,0.05);
                   padding:3px 10px;border-radius:100px;">RAG Audit</span>
      <span style="font-size:11px;color:#94A3B8;background:rgba(255,255,255,0.05);
                   padding:3px 10px;border-radius:100px;">Safety Trace</span>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

# ── ARCHITECTURE SECTION ────────────────────────────────────────
st.markdown("""
<div style="background:rgba(255,255,255,0.02); border:1px solid rgba(0,212,255,0.1);
            border-radius:16px; padding:28px 32px; margin-bottom:32px; animation:fadeIn 1s ease;">
  <div style="font-family:'Space Grotesk',sans-serif; font-size:13px; font-weight:600;
              color:#00D4FF; text-transform:uppercase; letter-spacing:2px; margin-bottom:16px;">
    Platform Architecture
  </div>
  <div style="display:flex; align-items:center; flex-wrap:wrap; gap:0;
              font-family:'Inter',sans-serif; font-size:12px;">

    <div style="text-align:center; padding:10px 16px;">
      <div style="color:#F1F5F9; font-weight:500; margin-bottom:2px;">PDF / Input</div>
      <div style="color:#475569; font-size:11px;">User data</div>
    </div>
    <div style="color:#00D4FF; font-size:18px; padding:0 4px;">→</div>
    <div style="text-align:center; padding:10px 16px;">
      <div style="color:#F1F5F9; font-weight:500; margin-bottom:2px;">Router Bot</div>
      <div style="color:#475569; font-size:11px;">Intent detection</div>
    </div>
    <div style="color:#00D4FF; font-size:18px; padding:0 4px;">→</div>
    <div style="text-align:center; padding:10px 16px;">
      <div style="color:#F1F5F9; font-weight:500; margin-bottom:2px;">RAG Retrieval</div>
      <div style="color:#475569; font-size:11px;">Vector search</div>
    </div>
    <div style="color:#00D4FF; font-size:18px; padding:0 4px;">→</div>
    <div style="text-align:center; padding:10px 16px;">
      <div style="color:#F1F5F9; font-weight:500; margin-bottom:2px;">Specialist Bot</div>
      <div style="color:#475569; font-size:11px;">Domain expert</div>
    </div>
    <div style="color:#00D4FF; font-size:18px; padding:0 4px;">→</div>
    <div style="text-align:center; padding:10px 16px;">
      <div style="color:#F1F5F9; font-weight:500; margin-bottom:2px;">Safety Check</div>
      <div style="color:#475569; font-size:11px;">Guardrails</div>
    </div>
    <div style="color:#00D4FF; font-size:18px; padding:0 4px;">→</div>
    <div style="text-align:center; padding:10px 16px;">
      <div style="color:#F1F5F9; font-weight:500; margin-bottom:2px;">Response</div>
      <div style="color:#475569; font-size:11px;">Plain language</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── QUICK-START ─────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:32px;">
  <div style="font-family:'Space Grotesk',sans-serif; font-size:20px; font-weight:600;
              color:#F1F5F9; margin-bottom:16px;">Quick Start</div>
  <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px;">
    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(0,212,255,0.1);
                border-radius:12px; padding:16px; display:flex; gap:14px; align-items:flex-start;">
      <div style="background:rgba(0,212,255,0.1); border-radius:8px; width:32px; height:32px;
                  display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:14px;">1</div>
      <div>
        <div style="color:#E2E8F0;font-size:13px;font-weight:500;margin-bottom:3px;">Set API Key</div>
        <div style="color:#64748B;font-size:12px;">Add OPENAI_API_KEY to .streamlit/secrets.toml</div>
      </div>
    </div>
    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(0,212,255,0.1);
                border-radius:12px; padding:16px; display:flex; gap:14px; align-items:flex-start;">
      <div style="background:rgba(0,212,255,0.1); border-radius:8px; width:32px; height:32px;
                  display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:14px;">2</div>
      <div>
        <div style="color:#E2E8F0;font-size:13px;font-weight:500;margin-bottom:3px;">Choose a Module</div>
        <div style="color:#64748B;font-size:12px;">Select from the sidebar navigation</div>
      </div>
    </div>
    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(0,212,255,0.1);
                border-radius:12px; padding:16px; display:flex; gap:14px; align-items:flex-start;">
      <div style="background:rgba(0,212,255,0.1); border-radius:8px; width:32px; height:32px;
                  display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:14px;">3</div>
      <div>
        <div style="color:#E2E8F0;font-size:13px;font-weight:500;margin-bottom:3px;">Generate or Chat</div>
        <div style="color:#64748B;font-size:12px;">Run the pipeline or upload a medical PDF</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── DISCLAIMER ──────────────────────────────────────────────────
st.markdown("""
<div style="background:rgba(245,158,11,0.05); border:1px solid rgba(245,158,11,0.2);
            border-radius:12px; padding:16px 20px; margin-top:8px;
            display:flex; gap:12px; align-items:flex-start;">
  <span style="font-size:18px; flex-shrink:0; margin-top:1px;">⚠️</span>
  <div style="font-family:'Inter',sans-serif;">
    <div style="color:#F59E0B; font-size:13px; font-weight:600; margin-bottom:4px;">
      Research & Educational Use Only
    </div>
    <div style="color:#94A3B8; font-size:12px; line-height:1.6;">
      MediExplain generates and explains <strong style="color:#E2E8F0;">synthetic, fictional</strong>
      medical records for research and educational purposes only. All patient data is
      AI-generated and entirely fictional. This platform is <strong style="color:#E2E8F0;">not a substitute
      for professional medical advice</strong>, diagnosis, or treatment. Never make clinical decisions
      based on AI-generated content. Always consult a qualified healthcare professional.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
