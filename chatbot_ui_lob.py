import os
import io
import re
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from ydata_profiling import ProfileReport

# Monkey-patch pandas to avoid tabulate dependency
pd.DataFrame.to_markdown = lambda self, **kwargs: self.to_string()

from langchain_openai.chat_models import AzureChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType

try:
    import fitz  # PyMuPDF
    has_pymupdf = True
except ImportError:
    has_pymupdf = False
    print("Warning: PyMuPDF not installed. PDF processing disabled.")

# ── Environment setup ─────────────────────────────────────────────────────────
load_dotenv()
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# ── LLM client ─────────────────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.1,
    streaming=False,
)

# ── Preload datasets ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
inc1 = pd.read_csv(os.path.join(DATA_DIR, "bus_incidents.csv"))
inc2 = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
bus_df = inc1.merge(inc2, left_on="case_no", right_on="bocc_number", how="outer")
personal_df = pd.read_csv(os.path.join(DATA_DIR, "personal_accidents.csv"))
PRELOADED = {
    "Bus Incidents (merged)": bus_df,
    "Personal Accidents": personal_df
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_text(path: str) -> str:
    ext = path.lower().split('.')[-1]
    if ext == "pdf" and has_pymupdf:
        try:
            doc = fitz.open(path)
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            return f"Error extracting text from PDF {path}: {e}"
    try:
        return open(path, "r", errors="ignore").read()
    except Exception as e:
        return f"Error reading file {path}: {e}"

# Prompt templates & validation
PROMPTS = {
    "list": "Provide a concise list for the following request:",
    "explain": "Explain step-by-step the following:",
    "default": "Answer the following question:"
}

def get_question_type(q: str) -> str:
    ql = q.lower().strip()
    if ql.startswith(("list", "show", "what are", "give me", "top", "find")):
        return "list"
    if any(w in ql for w in ["explain", "why", "how"]):
        return "explain"
    return "default"

def build_prompt(q: str) -> str:
    return f"{PROMPTS[get_question_type(q)]} {q}"

NUMERIC_PATTERNS = r"\b(how many|count|number of|average|sum|total)\b"

def validate_answer(question: str, answer: str) -> (str, str):
    if re.search(NUMERIC_PATTERNS, question.lower()) and not re.search(r"\d", answer):
        corrected = llm.predict(f"Q: {question}\nYour answer: {answer}. Please include a numeric value.")
        return corrected.strip(), "Answer self-corrected to include numeric value."
    return answer, ""

last_interaction = {"question": None, "answer": None}
rating_store = []

# ── Gradio UI ───────────────────────────────────────────────────────────────────
css = """
.gr-button { max-width: 100px !important; font-size: 0.85rem !important; }
.gr-radio, .gr-dropdown, .gr-checkbox, .gr-textbox { font-size: 0.9rem !important; }
.sample-btn { margin: 2px; font-size: 0.75rem !important; padding: 4px 8px !important; }
"""
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Intelligent Document Assistant")

    query_in = gr.Textbox(lines=2, label="Your Question")
    with gr.Accordion("Sample Questions", open=False):
        samples = [
            "What are the top observations from the dataset?",
            "How many accidents occurred last month?",
            "Show top 5 accident causes by zone.",
            "Explain trend of personal accidents over time.",
            "Generate a histogram of case durations."
        ]
        with gr.Row():
            for s in samples:
                gr.Button(s, elem_classes="sample-btn").click(lambda s=s: s, outputs=[query_in])

    gr.Markdown("---")
    mode_select = gr.Radio([
        "Preloaded Data", "Upload CSV", "Upload PDF/TXT", "General Question"
    ], label="Select Mode", value="Preloaded Data")
    preload_select = gr.Dropdown(list(PRELOADED.keys()), label="Choose Dataset")
    file_upload = gr.File(label="Upload CSV", file_count="multiple", visible=False)
    is_plot_q = gr.Checkbox(label="This question requires a plot", visible=False)
    with gr.Row(visible=False) as eda_controls:
        eda_btn = gr.Button("Generate EDA")
        clear_eda = gr.Button("Clear EDA")
    with gr.Accordion("EDA Report", open=False, visible=False) as eda_acc:
        eda_output = gr.HTML()
    with gr.Row():
        submit_btn = gr.Button("Submit")
        plot_btn = gr.Button("Plot")
        insights_btn = gr.Button("Insights")
        clear_btn = gr.Button("Clear All")
    answer_md = gr.Markdown()
    plot_img = gr.Image(visible=False, type="pil")
    with gr.Row():
        up_btn = gr.Button("👍")
        down_btn = gr.Button("👎")
        download_btn = gr.File(label="Download Output", visible=False)
    notes_tb = gr.Textbox(label="Notes", interactive=False)

    def toggle_ui(m):
        pre = (m=="Preloaded Data"); upc = (m=="Upload CSV")
        return (
            gr.update(visible=pre),
            gr.update(visible=upc),
            gr.update(visible=upc),
            gr.update(visible=pre or upc),
            gr.update(visible=pre or upc)
        )
    mode_select.change(
        toggle_ui,
        inputs=[mode_select],
        outputs=[preload_select, file_upload, is_plot_q, eda_controls, eda_acc]
    )

    clear_btn.click(
        lambda: ("", None, None, ""),
        outputs=[query_in, answer_md, plot_img, notes_tb]
    )

    def save_output(text, has_plot=False):
        fn = f"{'plot_output' if has_plot else 'output'}_{datetime.now():%Y%m%d_%H%M%S}.txt"
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(text)
        return fn

    def gen_eda(mode, files, sel):
        df = PRELOADED[sel] if mode=="Preloaded Data" else (pd.read_csv(files[0].name) if files else pd.DataFrame())
        if df.empty:
            return "No data to profile.", gr.update(open=False)
        buf = io.StringIO()
        df.info(buf=buf)
        info_html = buf.getvalue().replace("\n", "<br>")
        stats_md = df.describe(include='all').transpose().to_markdown()
        profile_html = ProfileReport(df, minimal=True, explorative=True).to_html()
        html = (
            f"<h3>Info</h3><p>{info_html}</p>"
            f"<h3>Stats</h3><pre>{stats_md}</pre>" + profile_html
        )
        return html, gr.update(open=True)
    eda_btn.click(gen_eda, inputs=[mode_select, file_upload, preload_select], outputs=[eda_output, eda_acc])
    clear_eda.click(lambda: ("", gr.update(open=False)), outputs=[eda_output, eda_acc])

    def query_agent(mode, files, plot_flag, insights_flag, sel, q):
        if mode=="Preloaded Data":
            df = PRELOADED[sel]
            tmp = os.path.join(BASE_DIR, "__temp.csv")
            df.to_csv(tmp, index=False)
            files = [type('F',(),{'name':tmp})()]
        if mode=="Upload CSV" and not files:
            return "", gr.update(visible=False), gr.update(visible=False), "Upload a CSV."
        if plot_flag:
            sample = pd.read_csv(files[0].name).head().to_string()
            resp = llm.predict(f"Suggest best chart & generate code to answer: {q}. Sample:\n{sample}")
            match = re.search(r"```python\n(.*?)\n```", resp, re.DOTALL)
            if match:
                exec(match.group(1), {"pd": pd, "plt": plt})
                buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
                ans, img = "Plot generated.", Image.open(buf)
            else:
                ans, img = "No plotting code found.", None
            last_interaction.update({"question": q, "answer": ans})
            return ans, gr.update(value=img, visible=bool(img)), gr.update(visible=False), ""
        if insights_flag:
            df = PRELOADED[sel] if mode=="Preloaded Data" else pd.read_csv(files[0].name)
            sample_df = df.sample(min(len(df),50), random_state=42).head(10)
            sample_md = sample_df.to_markdown()
            stats = df.describe(percentiles=[.1,.5,.9]).transpose(); stats_md = stats.to_markdown()
            num_df = df.select_dtypes(include='number')
            if num_df.shape[1]>=2:
                corr_ser = num_df.corr().abs().stack()
                corr_ser = corr_ser[corr_ser.index.get_level_values(0)!=corr_ser.index.get_level_values(1)]
                top_corr = corr_ser.groupby(level=[0,1]).first().nlargest(3)
                corr_df = top_corr.reset_index(); corr_df.columns=["feature_A","feature_B","correlation"]
                corr_md = corr_df.to_markdown()
            else:
                corr_md = "No numeric columns for correlation."
            anomalies = {}
            for col in num_df.columns:
                mu, sigma = num_df[col].mean(), num_df[col].std()
                outs = num_df[(num_df[col]>mu+2*sigma)|(num_df[col]<mu-2*sigma)][col].unique().tolist()[:5]
                if outs: anomalies[col] = outs
            anomaly_lines = "\n".join(f"- **{c}**: {v}" for c, v in anomalies.items()) or "No anomalies detected."
            context = (
                "### Sample 10 of 50 rows:\n" + sample_md + "\n\n"
                "### Summary Statistics:\n" + stats_md + "\n\n"
                "### Top 3 Correlations:\n" + corr_md + "\n\n"
                "### Anomalies (±2σ):\n" + anomaly_lines + "\n\n"
            )
            llm_prompt = (
                context +
                "You are an analytics assistant. Provide narrative key findings, highlight trends & outliers, discuss correlations, and suggest next steps.\n" +
                f"Question: {q}"
            )
            ans = llm.predict(llm_prompt)
            last_interaction.update({"question": q, "answer": ans})
            path = save_output(ans)
            return ans, gr.update(visible=False), gr.update(value=path, visible=True), ""
        agent = create_csv_agent(llm, files[0].name, verbose=False, allow_dangerous_code=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
        ans = agent.run(build_prompt(q)); ans, _ = validate_answer(q, ans)
        last_interaction.update({"question": q, "answer": ans})
        path = save_output(ans)
        return ans, gr.update(visible=False), gr.update(value=path, visible=True), ""

    submit_btn.click(lambda m,f,p,sel,q: query_agent(m,f,p,False,sel,q), inputs=[mode_select,file_upload,is_plot_q,preload_select,query_in], outputs=[answer_md,plot_img,download_btn,notes_tb])
    plot_btn.click(lambda m,f,sel,q: query_agent(m,f,True,False,sel,q), inputs=[mode_select,file_upload,preload_select,query_in], outputs=[answer_md,plot_img,download_btn,notes_tb])
    insights_btn.click(lambda m,f,sel,q: query_agent(m,f,False,True,sel,q), inputs=[mode_select,file_upload,preload_select,query_in], outputs=[answer_md,plot_img,download_btn,notes_tb])

if __name__ == "__main__":
    # Use PORT env var for Azure, default to 7860 locally
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
