import streamlit as st
import pandas as pd
import base64
import os
import zipfile
from datetime import datetime
from playwright.sync_api import sync_playwright
from openai import OpenAI
from dotenv import load_dotenv
import io
from PIL import Image
import re
from urllib.parse import urlparse

# Load environment variables (optional, if user prefers .env)
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Visual QA Migration Tool", layout="wide")

if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# --- CSS for styling ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components

# --- Functions ---

def get_screenshot(url, selectors_to_hide, width=1440, credentials=None):
    """Captures a full-page screenshot after hiding specific selectors at a given width."""
    try:
        with sync_playwright() as p:
            # Try system Chromium first (for Streamlit Cloud), then fallback to Playwright's
            chromium_paths = [
                "/usr/bin/chromium",           # Streamlit Cloud / Linux
                "/usr/bin/chromium-browser",   # Alternative Linux path
                "/usr/bin/google-chrome",      # Google Chrome on Linux
                None                            # Playwright's bundled browser (local dev)
            ]
            
            browser = None
            for chrome_path in chromium_paths:
                try:
                    if chrome_path:
                        browser = p.chromium.launch(
                            headless=True, 
                            executable_path=chrome_path,
                            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
                        )
                    else:
                        browser = p.chromium.launch(headless=True)
                    break
                except Exception:
                    continue
            
            if not browser:
                return None, "Could not launch any browser"
            
            # Setup context with high-res viewport and optional credentials
            # Height is set to 1080 initially, but we take full_page=True anyway
            context_args = {
                'viewport': {'width': width, 'height': 1080},
                'ignore_https_errors': True
            }
            if credentials and credentials.get("username") and credentials.get("password"):
                context_args['http_credentials'] = credentials
                
            context = browser.new_context(**context_args)
            page = context.new_page()
            
            # Navigate to URL
            page.goto(url, wait_until='networkidle', timeout=60000)
            
            # Inject CSS to hide selectors
            if selectors_to_hide:
                selectors_list = [s.strip() for s in selectors_to_hide.split(',') if s.strip()]
                if selectors_list:
                    css = ", ".join(selectors_list) + " { display: none !important; }"
                    page.add_style_tag(content=css)
            
            # Take full page screenshot
            screenshot_bytes = page.screenshot(full_page=True)
            browser.close()
            return screenshot_bytes, None
    except Exception as e:
        return None, str(e)

def encode_image(image_bytes):
    """Encodes image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def crop_image(image_bytes, box_normalized):
    """Crops an image based on [ymin, xmin, ymax, xmax] normalized coordinates (0-1000)."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        
        # Scale coordinates
        ymin, xmin, ymax, xmax = box_normalized
        left = (xmin / 1000) * width
        top = (ymin / 1000) * height
        right = (xmax / 1000) * width
        bottom = (ymax / 1000) * height
        
        # Add some padding
        padding_w = (right - left) * 0.2
        padding_h = (bottom - top) * 0.2
        left = max(0, left - padding_w)
        top = max(0, top - padding_h)
        right = min(width, right + padding_w)
        bottom = min(height, bottom + padding_h)
        
        return img.crop((left, top, right, bottom))
    except Exception:
        return None

def clean_json_response(text):
    """Strips markdown code blocks and attempts to parse JSON."""
    if "PASS" in text.upper():
        return "PASS"
    
    # Try to find JSON block
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1)
    else:
        # Try to find anything between { } or [ ]
        match = re.search(r"([\\[\{][\s\S]*[\}\]])", text)
        if match:
            text = match.group(1)
            
    try:
        data = json.loads(text)
        # Normalize to a list of bugs
        if isinstance(data, list):
            return {"bugs": data}
        if isinstance(data, dict):
            if "bugs" in data:
                return data
            # If it's a single bug object, wrap it
            if "description" in data:
                return {"bugs": [data]}
        return data
    except Exception:
        return text

def get_report_file_info(res):
    """Returns (absolute_path, filename) for a report, creating subdirs as needed."""
    # 1. Base reports directory
    base_dir = os.path.join(os.getcwd(), "comparison_reports")
    
    # 2. Width-based subdirectory
    width_dir = os.path.join(base_dir, f"{res['width']}px")
    os.makedirs(width_dir, exist_ok=True)
    
    # 3. Clean path for filename
    target_url = res.get('test_url') or res.get('url')
    parsed = urlparse(target_url)
    # Remove leading/trailing slashes and replace others with underscores
    path_clean = parsed.path.strip("/").replace("/", "_")
    if not path_clean:
        path_clean = "root"

    if res.get('mode') == 'screenshot':
        filename = f"{path_clean}_screenshot_{res['width']}px.html"
    else:
        filename = f"{path_clean}_report_{res['width']}px.html"

    full_path = os.path.join(width_dir, filename)
    
    return full_path, filename

def create_reports_archive(results):
    """Creates a ZIP archive with all HTML reports and returns bytes for download."""
    # Create a BytesIO object to store the zip file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for res in results:
            # For compare mode, errors are stored as prod_err/test_err.
            # For screenshot-only, we store err in 'err' and keep prod_err/test_err empty.
            if not res.get('prod_err') and not res.get('test_err') and not res.get('err'):
                # Generate HTML content
                if res.get('mode') == 'screenshot':
                    report_html = generate_screenshot_only_html(res)
                    parsed = urlparse(res['url'])
                    path_clean = parsed.path.strip("/").replace("/", "_")
                    if not path_clean:
                        path_clean = "root"
                    filename = f"{res['width']}px/{path_clean}_screenshot_{res['width']}px.html"
                else:
                    report_html = generate_comparison_html(res)
                    parsed = urlparse(res['test_url'])
                    path_clean = parsed.path.strip("/").replace("/", "_")
                    if not path_clean:
                        path_clean = "root"
                    filename = f"{res['width']}px/{path_clean}_report_{res['width']}px.html"

                # Add to zip
                zip_file.writestr(filename, report_html)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def generate_comparison_html(res):
    """Generates a standalone HTML string for the comparison pair with sticky headers."""
    prod_b64 = encode_image(res['prod_img'])
    test_b64 = encode_image(res['test_img'])
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>QA Comparison: #{res['index']}</title>
        <style>
            body {{ 
                margin: 0; 
                padding: 0 20px 20px 20px; 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; 
                background-color: #f0f2f6;
                color: #333;
            }}
            .container {{ 
                display: flex; 
                gap: 20px; 
                align-items: flex-start;
                max-width: 100%;
                padding-top: 0;
            }}
            .column {{ 
                flex: 1; 
                min-width: 0; 
                background: white;
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            }}
            .header {{
                position: -webkit-sticky;
                position: sticky;
                top: 0;
                z-index: 1000;
                color: white;
                padding: 14px 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                font-weight: 700;
                font-size: 14px;
                backdrop-filter: blur(5px);
                -webkit-backdrop-filter: blur(5px);
            }}
            .prod-header {{ background: rgba(255, 75, 75, 0.95); }}
            .test-header {{ background: rgba(0, 102, 204, 0.95); }}
            .header a {{ 
                color: white; 
                text-decoration: none; 
                display: block; 
                margin-top: 4px; 
                font-weight: 400; 
                opacity: 0.95; 
                font-size: 13px;
                word-break: break-all;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}
            .header a:hover {{ text-decoration: underline; opacity: 1; }}
            img {{ width: 100%; display: block; }}
            .label {{ 
                font-size: 10px; 
                opacity: 0.8; 
                text-transform: uppercase; 
                letter-spacing: 1.5px; 
                margin-bottom: 2px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="column">
                <div class="header prod-header">
                    <div class="label">Production</div>
                    <a href="{res['prod_url']}" target="_blank">{res['prod_url']}</a>
                </div>
                <img src="data:image/png;base64,{prod_b64}">
            </div>
            <div class="column">
                <div class="header test-header">
                    <div class="label">Test (Migration)</div>
                    <a href="{res['test_url']}" target="_blank">{res['test_url']}</a>
                </div>
                <img src="data:image/png;base64,{test_b64}">
            </div>
        </div>
    </body>
    </html>
    """
    return html


def generate_screenshot_only_html(res):
    """Generates a standalone HTML string for a single URL screenshot with sticky header."""
    img_b64 = encode_image(res['img'])
    url = res['url']

    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\">
        <title>QA Screenshot: #{res['index']}</title>
        <style>
            body {{
                margin: 0;
                padding: 0 20px 20px 20px;
                font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Helvetica, Arial, sans-serif;
                background-color: #f0f2f6;
                color: #333;
            }}
            .wrap {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                overflow: hidden;
            }}
            .header {{
                position: -webkit-sticky;
                position: sticky;
                top: 0;
                z-index: 1000;
                background: rgba(0, 102, 204, 0.95);
                color: white;
                padding: 14px 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                font-weight: 700;
                font-size: 14px;
                backdrop-filter: blur(5px);
                -webkit-backdrop-filter: blur(5px);
            }}
            .label {{
                font-size: 10px;
                opacity: 0.8;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                margin-bottom: 2px;
            }}
            .header a {{
                color: white;
                text-decoration: none;
                display: block;
                margin-top: 4px;
                font-weight: 400;
                opacity: 0.95;
                font-size: 13px;
                word-break: break-all;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}
            .header a:hover {{ text-decoration: underline; opacity: 1; }}
            img {{ width: 100%; display: block; }}
        </style>
    </head>
    <body>
        <div class=\"wrap\">
            <div class=\"header\">
                <div class=\"label\">Screenshot</div>
                <a href=\"{url}\" target=\"_blank\">{url}</a>
            </div>
            <img src=\"data:image/png;base64,{img_b64}\">
        </div>
    </body>
    </html>
    """
    return html

def analyze_images(img_bytes_prod, img_bytes_test, api_key, model_name="gpt-4o"):
    """Sends production and test images to GPT for visual comparison with coordinates."""
    if not api_key:
        return "Error: OpenAI API Key is missing."
    
    client = OpenAI(api_key=api_key)
    
    base64_prod = encode_image(img_bytes_prod)
    base64_test = encode_image(img_bytes_test)
    
    system_prompt = (
        "You are a Senior QA Automation Engineer. Compare the 'Production' image (A) with the 'Test' image (B).\n"
        "Your goal is content validation, not pixel-perfect layout comparison.\n\n"
        "CRITICAL CHECKS:\n"
        "1. TEXT CONTENT: Verify all text (headings, body, labels) is identical in meaning and spelling. Flag any typos or missing text.\n"
        "2. FONTS & STYLING: Check if fonts for headings and body look consistent. Flag obvious font-family mismatches.\n"
        "3. IMAGES: Verify presence of images. Do not compare image content/details, just check if an image exists where it should.\n"
        "4. MISSING SECTIONS: Flag if any blocks or sections from A are completely missing in B.\n\n"
        "IGNORE:\n"
        "- Minor 1-5px shifts, padding differences, or small alignment tweaks.\n"
        "- Rendering artifacts or slight color shade differences.\n\n"
        "OUTPUT FORMAT: Provide a JSON object with a key 'bugs' containing a list. "
        "Each bug must have 'description' and 'box_2d' [ymin, xmin, ymax, xmax] "
        "normalized to 0-1000 representing the area in Image B (Test).\n"
        "If the content and feeling are effectively the same, reply exactly with 'PASS'."
    )
    
    try:
        response = client.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these two screenshots. Image A is Production, Image B is Test. Provide bug locations in Image B."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_prod}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_test}"}}
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during AI analysis: {str(e)}"

# --- Sidebar UI ---
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Viewport Configuration")
    selected_widths = st.multiselect("Select Widths (px)", 
                                    options=[1440, 1024, 768, 375], 
                                    default=[1440],
                                    help="Select one or more widths to test responsiveness.")
    
    st.markdown("---")
    default_selectors = "#onetrust-banner-sdk, .intercom-lightweight-app, iframe, #chat-widget-container, header, footer"
    selectors_to_hide = st.text_area("CSS Selectors to Hide", value=default_selectors, help="Comma-separated list of CSS selectors to hide before taking screenshots.")
    
    st.markdown("---")
    st.subheader("Test Env Credentials")
    st.caption("Optional: Use if your test environment is behind Basic Auth.")
    test_user = st.text_input("Test Username", value="", help="User for Basic Auth").strip()
    test_pass = st.text_input("Test Password", type="password", value="", help="Password for Basic Auth").strip()
    
    st.markdown("---")
    st.subheader("Run Mode")
    run_mode = st.radio(
        "Choose mode",
        options=["Compare (prod vs test)", "Screenshots only (single list)"],
        index=0,
        help="Compare requires prod_url + test_url. Screenshots-only accepts a single column (with header or without) and produces HTML reports per URL."
    )

    st.markdown("---")
    uploader_label = "Upload CSV (prod_url, test_url)" if run_mode.startswith("Compare") else "Upload CSV (url list: header optional)"
    uploaded_file = st.file_uploader(uploader_label, type=["csv"])
    
    if st.session_state.is_running:
        if st.button("Stop Run"):
            st.session_state.is_running = False
            st.rerun()
    else:
        start_label = "Start Comparison" if run_mode.startswith("Compare") else "Start Screenshots"
        start_button = st.button(start_label)
    
    # AI Analysis section at the bottom
    st.markdown("---")
    with st.expander("🤖 AI Analysis (Optional)"):
        use_ai = st.checkbox("Enable AI Analysis", value=False, help="Use GPT to identify bugs automatically.")
        api_key = None
        model_name = "gpt-4o"
        if use_ai:
            api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
            model_name = st.selectbox("OpenAI Model", 
                                   options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], 
                                       index=0)

# --- Main Area ---
st.title("🔍 VisualDiff")
st.markdown("Compare production and test pages side-by-side. Capture full-page screenshots and spot visual differences instantly.")

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

if not st.session_state.is_running and 'start_button' in locals() and start_button:
    if use_ai and not api_key:
        st.error("Please provide an OpenAI API Key in the sidebar to use AI Analysis.")
    elif uploaded_file is None:
        st.error("Please upload a CSV file.")
    else:
        st.session_state.is_running = True
        st.rerun()

if st.session_state.is_running:
    try:
        # We need to re-read the file since we reran
        if uploaded_file is not None:
            # Read bytes once so we can re-parse reliably on reruns
            csv_bytes = uploaded_file.getvalue()

            def parse_urls_single_list(data: bytes):
                """Parse a CSV that may have a header or not, and may use column name 'url' or a single unnamed column."""
                # Attempt 1: header auto
                try:
                    df1 = pd.read_csv(io.BytesIO(data))
                except Exception:
                    df1 = pd.DataFrame()

                urls = []
                if not df1.empty:
                    if 'url' in df1.columns:
                        urls = df1['url'].tolist()
                    elif 'test_url' in df1.columns and 'prod_url' not in df1.columns:
                        # Sometimes people will upload a single-column file but keep the old name
                        urls = df1['test_url'].tolist()
                    elif df1.shape[1] == 1:
                        urls = df1.iloc[:, 0].tolist()

                # Detect the "first URL became header" case → retry with header=None
                if not urls:
                    try:
                        df2 = pd.read_csv(io.BytesIO(data), header=None)
                        if df2.shape[1] >= 1:
                            urls = df2.iloc[:, 0].tolist()
                    except Exception:
                        urls = []

                # Clean
                cleaned = []
                for u in urls:
                    if pd.isna(u):
                        continue
                    u = str(u).strip()
                    if not u:
                        continue
                    cleaned.append(u)
                return cleaned

            # Prepare credentials if provided (strip just in case)
            test_creds = None
            if test_user and test_pass:
                test_creds = {"username": test_user, "password": test_pass}

            st.session_state.results = []  # Clear previous results
            st.session_state.current_index = 0

            if run_mode.startswith("Compare"):
                df = pd.read_csv(io.BytesIO(csv_bytes))
                if 'prod_url' not in df.columns or 'test_url' not in df.columns:
                    st.error("CSV must contain 'prod_url' and 'test_url' columns for Compare mode.")
                    st.session_state.is_running = False
                else:
                    total_urls = len(df)
                    total_ops = total_urls * len(selected_widths)
                    current_op = 0

                    progress_bar = st.progress(0)
                    progress_status = st.empty()

                    for index, row in df.iterrows():
                        if not st.session_state.is_running:
                            break

                        prod_url = str(row['prod_url']).strip()
                        test_url = str(row['test_url']).strip()

                        for width in selected_widths:
                            if not st.session_state.is_running:
                                break
                        
                        current_op += 1
                        progress_percentage = current_op / total_ops
                        
                        # Update status with clear metrics
                        progress_status.markdown(f"""
                        **Progress:** {current_op} / {total_ops} operations completed
                        *   **Current URL:** {index + 1} of {total_urls}
                        *   **Current Viewport:** `{width}px`
                        *   **Processing:** `{test_url}`
                        """)
                        progress_bar.progress(progress_percentage)
                        
                        # Capture Screenshots
                        prod_img, prod_err = get_screenshot(prod_url, selectors_to_hide, width=width)
                        test_img, test_err = get_screenshot(test_url, selectors_to_hide, width=width, credentials=test_creds)
                        
                        res_item = {
                            "mode": "compare",
                            "index": index + 1,
                            "width": width,
                            "prod_url": prod_url,
                            "test_url": test_url,
                            "prod_img": prod_img,
                            "test_img": test_img,
                            "prod_err": prod_err,
                            "test_err": test_err,
                            "ai_result": None
                        }

                        if not prod_err and not test_err:
                            # AI Analysis
                            if use_ai:
                                raw_analysis = analyze_images(prod_img, test_img, api_key, model_name=model_name)
                                res_item["ai_result"] = clean_json_response(raw_analysis)
                        
                        st.session_state.results.append(res_item)
                
                progress_status.empty()
                progress_bar.empty()
                st.session_state.is_running = False
                
                st.success("Comparison completed! Use the sidebar to navigate results.")

            else:
                # --- Screenshots-only mode ---
                urls = parse_urls_single_list(csv_bytes)
                if not urls:
                    st.error("CSV must contain a single column of URLs (header optional). Example: a column named 'url' or just one column.")
                    st.session_state.is_running = False
                else:
                    total_urls = len(urls)
                    total_ops = total_urls * len(selected_widths)
                    current_op = 0

                    progress_bar = st.progress(0)
                    progress_status = st.empty()

                    for idx, url in enumerate(urls):
                        if not st.session_state.is_running:
                            break

                        for width in selected_widths:
                            if not st.session_state.is_running:
                                break

                            current_op += 1
                            progress_percentage = current_op / total_ops

                            progress_status.markdown(f"""
                            **Progress:** {current_op} / {total_ops} operations completed
                            *   **Current URL:** {idx + 1} of {total_urls}
                            *   **Current Viewport:** `{width}px`
                            *   **Processing:** `{url}`
                            """)
                            progress_bar.progress(progress_percentage)

                            img, err = get_screenshot(url, selectors_to_hide, width=width, credentials=test_creds)

                            res_item = {
                                "mode": "screenshot",
                                "index": idx + 1,
                                "width": width,
                                "url": url,
                                "img": img,
                                "err": err,
                                # Keep keys for compatibility with the rest of the UI
                                "prod_err": None,
                                "test_err": None,
                                "ai_result": None,
                            }
                            st.session_state.results.append(res_item)

                    progress_status.empty()
                    progress_bar.empty()
                    st.session_state.is_running = False

                    st.success("Screenshots completed! Use the sidebar to navigate results.")

        else:
            st.session_state.is_running = False
                
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.session_state.is_running = False

# --- Navigation & Display ---
if st.session_state.results:
    st.markdown("---")
    
    # Create navigation labels with width
    def _nav_label(r):
        target = r.get('test_url') or r.get('url')
        tail = target.split('/')[-1] if target else ""
        return f"#{r['index']} [{r['width']}px]: {tail or target}"

    nav_options = [_nav_label(r) for r in st.session_state.results]
    
    with st.sidebar:
        st.markdown("---")
        st.subheader("Results Navigation")
        
        # Download Archive Button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"qa_reports_{timestamp}.zip"
        
        archive_bytes = create_reports_archive(st.session_state.results)
        st.download_button(
            label="📦 Download All Reports (ZIP)",
            data=archive_bytes,
            file_name=archive_name,
            mime="application/zip",
            use_container_width=True
        )

        selected_nav = st.radio("Select Result", nav_options, index=st.session_state.current_index)
        st.session_state.current_index = nav_options.index(selected_nav)

    # Display selected result
    res = st.session_state.results[st.session_state.current_index]
    
    st.header(f"Result #{res['index']}")
    
    if res.get('err') or res.get('prod_err') or res.get('test_err'):
        if res.get('err'): st.error(f"Error: {res['err']}")
        if res.get('prod_err'): st.error(f"Prod Error: {res['prod_err']}")
        if res.get('test_err'): st.error(f"Test Error: {res['test_err']}")
    else:
        # Get path and filename using the new helper
        report_path, report_filename = get_report_file_info(res)

        # Display action area
        st.markdown(f"""
            <div style="background: white; padding: 25px; border-radius: 12px; border: 1px solid #eee; text-align: center; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #333;">Report Preview</h3>
                <p style="color: #666; font-size: 0.9em;">Download this report or all reports from the sidebar.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Download single report button
        report_html = generate_screenshot_only_html(res) if res.get('mode') == 'screenshot' else generate_comparison_html(res)
        st.download_button(
            label="📄 Download This Report",
            data=report_html,
            file_name=report_filename,
            mime="text/html",
            use_container_width=True
        )

        if use_ai:
            st.markdown("#### AI QA Feedback:")
            analysis_result = res['ai_result']
            if isinstance(analysis_result, str) and "PASS" in analysis_result:
                st.success(analysis_result)
            elif isinstance(analysis_result, dict) and "bugs" in analysis_result:
                bugs = analysis_result.get("bugs", [])
                if not bugs:
                    st.success("PASS (No bugs identified)")
                else:
                    for bug in bugs:
                        with st.container():
                            b_col1, b_col2 = st.columns([2, 1])
                            with b_col1:
                                st.error(f"**Bug:** {bug.get('description')}")
                            with b_col2:
                                box = bug.get("box_2d")
                                if box and len(box) == 4:
                                    snippet = crop_image(res['test_img'], box)
                                    if snippet:
                                        st.image(snippet, caption="Problem Area", width=250)
            else:
                st.error(f"Raw Analysis Output: {analysis_result}")
        else:
            st.info("AI Analysis was disabled for this run.")

elif uploaded_file is None:
        st.info("Upload a CSV to begin.")
        
        st.markdown("""
        **Compare mode (prod vs test):**
        ```csv
        prod_url,test_url
        https://example.com/page1,https://test.example.com/page1
        https://example.com/page2,https://test.example.com/page2
        ```

        **Screenshots-only mode (single list):**

        With header:
        ```csv
        url
        https://test.example.com/page1
        https://test.example.com/page2
        ```

        Without header:
        ```csv
        https://test.example.com/page1
        https://test.example.com/page2
        ```
        """)
