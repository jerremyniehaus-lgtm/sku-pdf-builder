import io
import time
import requests
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# =========================
# HARD SETTINGS
# =========================
BRAND_OPTIONS = {
    "Jared": "jared",
    "Kay": "kay",
    "Zales": "zales",
}

GRID_ROWS = 6
GRID_COLS = 4

IMG_SIZE_PX = 380
TEXT_BLOCK_HEIGHT = 150

BORDER_PX = 3
CELL_PAD = 10

SLEEP_SECONDS = 0.12

FONT_SKU = 30
FONT_LINE = 24
FONT_RANK = 26
FONT_PLACEHOLDER = 34


# =========================
# UI Styling
# =========================
def inject_css():
    st.markdown(
        """
        <style>
          /* ===== Page sizing + fix clipped header ===== */
          .block-container {
            max-width: 1040px;
            padding-top: 2.2rem !important;
            padding-bottom: 1.6rem !important;
          }

          /* Reduce random Streamlit whitespace */
          section.main > div { padding-top: 0rem !important; }

          /* Tighten vertical gaps in general */
          div[data-testid="stVerticalBlock"] { gap: 0.75rem; }

          /* ===== Title area ===== */
          .app-header {
            background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(16,185,129,0.12));
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 16px 18px 14px 18px;
            margin-bottom: 0.9rem;
          }

          .app-title {
            font-size: 2.05rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.15;
            margin: 0;
            padding: 0;
          }

          .app-subtitle {
            margin-top: 6px;
            color: rgba(255,255,255,0.72);
            font-size: 0.98rem;
          }

          /* ===== Cards ===== */
          .card {
            background: rgba(255,255,255,0.055);
            border: 1px solid rgba(255,255,255,0.11);
            border-radius: 16px;
            padding: 14px 14px;
            margin: 0;
          }

          .card-title {
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 10px;
            color: rgba(255,255,255,0.88);
          }

          .muted {
            color: rgba(255,255,255,0.68);
            font-size: 0.92rem;
          }

          .divider {
            height: 1px;
            background: rgba(255,255,255,0.10);
            margin: 12px 0;
          }

          /* ===== Status banners (no overlap, guaranteed spacing) ===== */
          .status-ok {
            display: block;
            background: rgba(16,185,129,0.12);
            border: 1px solid rgba(16,185,129,0.28);
            border-radius: 14px;
            padding: 10px 12px;
            margin-top: 10px;
            margin-bottom: 10px;
          }

          .status-warn {
            display: block;
            background: rgba(245,158,11,0.12);
            border: 1px solid rgba(245,158,11,0.28);
            border-radius: 14px;
            padding: 10px 12px;
            margin-top: 10px;
            margin-bottom: 10px;
          }

          /* ===== Upload box polish ===== */
          div[data-testid="stFileUploader"] section {
            border: 1px dashed rgba(255,255,255,0.28);
            border-radius: 14px;
            padding: 8px 10px 10px 10px;
            background: rgba(255,255,255,0.03);
          }

          /* ===== Buttons polish ===== */
          div.stButton > button {
            width: 100%;
            border-radius: 14px;
            padding: 0.78rem 1rem;
            font-weight: 750;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(59,130,246,0.90);
          }
          div.stButton > button:hover {
            filter: brightness(1.05);
            border: 1px solid rgba(255,255,255,0.20);
          }

          /* Download button */
          div[data-testid="stDownloadButton"] > button {
            width: 100%;
            border-radius: 14px;
            padding: 0.72rem 1rem;
            font-weight: 750;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(16,185,129,0.88);
          }
          div[data-testid="stDownloadButton"] > button:hover {
            filter: brightness(1.05);
            border: 1px solid rgba(255,255,255,0.20);
          }

          /* Make selectbox + toggle look cleaner */
          div[data-testid="stSelectbox"] > div {
            border-radius: 14px;
          }
          div[data-testid="stToggleSwitch"] {
            padding-top: 0.4rem;
          }

          /* Footer */
          .footer {
            margin-top: 12px;
            color: rgba(255,255,255,0.55);
            font-size: 0.85rem;
            text-align: center;
          }

          /* Prevent accidental margin collapse between markdown blocks */
          .no-collapse { display: block; height: 0.01px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def brand_icon(label):
    if label == "Jared":
        return "💎"
    if label == "Kay":
        return "✨"
    if label == "Zales":
        return "💍"
    return "🛍️"


# =========================
# URL builders
# =========================
def build_image_url(brand_domain, sku_numeric):
    return f"https://www.{brand_domain}.com/productimages/processed/V-{sku_numeric}_0_800.jpg?pristine=true"


def build_product_url(brand_domain, sku_numeric):
    return f"https://www.{brand_domain}.com/p/V-{sku_numeric}"


# =========================
# Helpers
# =========================
def normalize_sku(value):
    s = str(value).strip()
    if s.lower().startswith("v-"):
        s = s[2:]
    return s


def download_image(url, timeout=20):
    try:
        return requests.get(url, timeout=timeout)
    except Exception:
        return None


def get_font(size=18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def parse_money_to_float(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "":
        return None
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def format_money(val, decimals=0):
    try:
        return f"${float(val):,.{decimals}f}"
    except Exception:
        return ""


def format_int(val):
    try:
        return f"{int(float(val)):,}"
    except Exception:
        return ""


def make_missing_image_block(message, size_px, placeholder_font_size):
    img = Image.new("RGB", (size_px, size_px), "white")
    draw = ImageDraw.Draw(img)

    font = get_font(placeholder_font_size)
    lines = message.split("\n")
    line_height = placeholder_font_size + 10
    total_h = len(lines) * line_height

    y = (size_px - total_h) // 2
    for line in lines:
        w = draw.textlength(line, font=font)
        x = (size_px - w) // 2
        draw.text((x, y), line, fill="black", font=font)
        y += line_height

    draw.rectangle([(0, 0), (size_px - 1, size_px - 1)], outline="black", width=3)
    return img


def build_items_from_df(df, brand_domain):
    if "SKU" not in df.columns:
        raise ValueError('Missing required column: "SKU"')
    if "Revenue" not in df.columns:
        raise ValueError('Missing required column: "Revenue" (needed to compute Rank)')

    df = df.copy()
    df["SKU_clean"] = df["SKU"].apply(normalize_sku)
    df = df.dropna(subset=["SKU_clean"])
    df = df[df["SKU_clean"] != ""]
    df = df.drop_duplicates(subset=["SKU_clean"], keep="first")

    df["Revenue_num"] = df["Revenue"].apply(parse_money_to_float)

    if "AUR" in df.columns:
        df["AUR_num"] = df["AUR"].apply(parse_money_to_float)
    else:
        df["AUR_num"] = None

    mask = df["Revenue_num"].notna()
    df["Rank"] = None
    df.loc[mask, "Rank"] = df.loc[mask, "Revenue_num"].rank(ascending=False, method="dense").astype(int)

    items_all = []
    for _, row in df.iterrows():
        sku_num = row["SKU_clean"]
        sku_label = f"V-{sku_num}"

        revenue_str = format_money(row["Revenue_num"], decimals=0) if pd.notna(row["Revenue_num"]) else ""

        units_str = ""
        if "Units" in df.columns and pd.notna(row.get("Units")):
            units_str = format_int(row.get("Units"))

        aur_str = ""
        if pd.notna(row.get("AUR_num")):
            aur_str = format_money(row.get("AUR_num"), decimals=0)

        rank_str = ""
        if row.get("Rank") is not None and not pd.isna(row.get("Rank")):
            rank_str = str(int(row.get("Rank")))

        items_all.append(
            {
                "sku_label": sku_label,
                "img_url": build_image_url(brand_domain, sku_num),
                "product_url": build_product_url(brand_domain, sku_num),
                "revenue_str": revenue_str,
                "units_str": units_str,
                "aur_str": aur_str,
                "rank_str": rank_str,
            }
        )

    return items_all


def make_tile_image(item):
    img_url = item["img_url"]
    product_url = item["product_url"]

    response = download_image(img_url)
    status_code = response.status_code if response is not None else None

    if response is None:
        img = make_missing_image_block("Image\nUnavailable", IMG_SIZE_PX, FONT_PLACEHOLDER)
    elif status_code == 404:
        img = make_missing_image_block("No Longer\nAvailable", IMG_SIZE_PX, FONT_PLACEHOLDER)
    elif status_code != 200:
        img = make_missing_image_block("Image\nUnavailable", IMG_SIZE_PX, FONT_PLACEHOLDER)
    else:
        try:
            img_data = io.BytesIO(response.content)
            img = Image.open(img_data).convert("RGB")
            img = img.resize((IMG_SIZE_PX, IMG_SIZE_PX))
        except Exception:
            img = make_missing_image_block("Image\nUnavailable", IMG_SIZE_PX, FONT_PLACEHOLDER)

    tile_w = IMG_SIZE_PX
    tile_h = IMG_SIZE_PX + TEXT_BLOCK_HEIGHT

    tile = Image.new("RGB", (tile_w, tile_h), "white")
    draw = ImageDraw.Draw(tile)

    tile.paste(img, (0, 0))

    f_sku = get_font(FONT_SKU)
    f_line = get_font(FONT_LINE)
    f_rank = get_font(FONT_RANK)

    line1 = item["sku_label"]
    line2 = f"Rev: {item['revenue_str']}"
    line3 = f"Units: {item['units_str']}   AUR: {item['aur_str']}"
    line4 = f"Rank: {item['rank_str']}"

    y = IMG_SIZE_PX + 6
    draw.text((10, y), line1, fill="black", font=f_sku)
    draw.text((10, y + 40), line2, fill="black", font=f_line)
    draw.text((10, y + 75), line3, fill="black", font=f_line)
    draw.text((10, y + 112), line4, fill="black", font=f_rank)

    time.sleep(SLEEP_SECONDS)
    return tile, product_url


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def build_clickable_pdf(pages_items):
    pdf_buffer = io.BytesIO()

    tile_w = IMG_SIZE_PX
    tile_h = IMG_SIZE_PX + TEXT_BLOCK_HEIGHT

    cell_w = tile_w + (BORDER_PX * 2) + (CELL_PAD * 2)
    cell_h = tile_h + (BORDER_PX * 2) + (CELL_PAD * 2)

    page_w = GRID_COLS * cell_w
    page_h = GRID_ROWS * cell_h

    c = canvas.Canvas(pdf_buffer, pagesize=(page_w, page_h))

    for page_tiles in pages_items:
        for idx, (tile_img, link_url) in enumerate(page_tiles):
            row = idx // GRID_COLS
            col = idx % GRID_COLS

            x0 = col * cell_w
            y0 = page_h - ((row + 1) * cell_h)

            c.setLineWidth(BORDER_PX)
            c.rect(x0, y0, cell_w, cell_h)

            paste_x = x0 + BORDER_PX + CELL_PAD
            paste_y = y0 + BORDER_PX + CELL_PAD

            img_reader = ImageReader(tile_img)
            c.drawImage(
                img_reader,
                paste_x,
                paste_y,
                width=tile_w,
                height=tile_h,
                preserveAspectRatio=True,
                mask="auto",
            )

            c.linkURL(
                link_url,
                (paste_x, paste_y, paste_x + tile_w, paste_y + tile_h),
                relative=0,
            )

        c.showPage()

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer


def build_preview_page_image(tile_images):
    tile_w = IMG_SIZE_PX
    tile_h = IMG_SIZE_PX + TEXT_BLOCK_HEIGHT

    cell_w = tile_w + (BORDER_PX * 2) + (CELL_PAD * 2)
    cell_h = tile_h + (BORDER_PX * 2) + (CELL_PAD * 2)

    page_w = GRID_COLS * cell_w
    page_h = GRID_ROWS * cell_h

    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)

    for idx, tile in enumerate(tile_images):
        row = idx // GRID_COLS
        col = idx % GRID_COLS

        x0 = col * cell_w
        y0 = row * cell_h

        draw.rectangle([(x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1)], outline="black", width=BORDER_PX)

        paste_x = x0 + BORDER_PX + CELL_PAD
        paste_y = y0 + BORDER_PX + CELL_PAD
        page.paste(tile, (paste_x, paste_y))

    return page


def build_template_bytes():
    df = pd.DataFrame(
        {
            "SKU": ["564104305", "253329407"],
            "Revenue": [123456, 98765],
            "Units": [120, 88],
            "AUR": [1029, 1122],
        }
    )
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="SKU_List")
    out.seek(0)
    return out.getvalue()


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="SKU Visual Analyzer", layout="centered")
inject_css()

st.markdown(
    """
    <div class="app-header">
      <div class="app-title">SKU Visual Analyzer</div>
      <div class="app-subtitle">Upload an XLSX to generate a clickable PDF grid of product images.</div>
    </div>
    <span class="no-collapse"></span>
    """,
    unsafe_allow_html=True,
)

# Top controls in a clean 2-column layout
top_left, top_right = st.columns([1.25, 1.0], gap="large")

with top_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Input file</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted"><b>Required:</b> SKU, Revenue<br/><b>Optional:</b> Units, AUR</div>', unsafe_allow_html=True)

    st.download_button(
        label="Download XLSX template",
        data=build_template_bytes(),
        file_name="SKU_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload XLSX", type=["xlsx"])
    st.markdown("</div>", unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Settings</div>', unsafe_allow_html=True)

    brand_label = st.selectbox(
        "Brand",
        list(BRAND_OPTIONS.keys()),
        index=0,
        format_func=lambda x: f"{brand_icon(x)}  {x}",
    )

    show_all_pages = st.toggle("Preview all pages", value=False)
    st.markdown("</div>", unsafe_allow_html=True)

brand_domain = BRAND_OPTIONS[brand_label]

# Main action area
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Generate</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown('<div class="muted">Upload an XLSX to begin.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    try:
        df = pd.read_excel(uploaded_file)
        items_all = build_items_from_df(df, brand_domain)

        per_page = GRID_ROWS * GRID_COLS
        total_pages = (len(items_all) + per_page - 1) // per_page

        st.markdown(
            f'<div class="status-ok"><b>Loaded:</b> {len(items_all)} unique SKUs &nbsp;&nbsp; <b>Pages:</b> {total_pages}</div>',
            unsafe_allow_html=True,
        )

        generate_clicked = st.button("Generate PDF + Preview", type="primary")

        st.markdown("</div>", unsafe_allow_html=True)

        if generate_clicked:
            progress = st.progress(0)
            status = st.empty()

            pages_tiles = []
            pages_preview_images = []

            processed = 0

            for page_num, chunk in enumerate(chunk_list(items_all, per_page), start=1):
                status.markdown(f"**Building page {page_num} of {total_pages}...**")

                page_tiles = []
                page_tile_images_for_preview = []

                for item in chunk:
                    tile_img, link_url = make_tile_image(item)
                    page_tiles.append((tile_img, link_url))
                    page_tile_images_for_preview.append(tile_img)

                    processed += 1
                    progress.progress(min(1.0, processed / len(items_all)))

                pages_tiles.append(page_tiles)

                if page_num == 1 or show_all_pages:
                    preview_img = build_preview_page_image(page_tile_images_for_preview)
                    pages_preview_images.append((page_num, preview_img))

            status.markdown("**Building PDF...**")
            pdf_buffer = build_clickable_pdf(pages_tiles)

            file_name = f"SKU_Visual_Analyzer_{brand_domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            st.markdown('<div class="status-ok"><b>Done.</b> Preview below, then download your PDF.</div>', unsafe_allow_html=True)

            st.download_button(
                label="Download PDF",
                data=pdf_buffer,
                file_name=file_name,
                mime="application/pdf",
            )

            st.markdown('<div class="card" style="margin-top: 0.85rem;">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Preview</div>', unsafe_allow_html=True)

            for page_num, preview_img in pages_preview_images:
                st.markdown(f"**Page {page_num}**")
                st.image(preview_img, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="status-warn"><b>Error:</b> {str(e)}</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="footer">Each PDF tile is clickable and opens the SKU product page.</div>', unsafe_allow_html=True)



