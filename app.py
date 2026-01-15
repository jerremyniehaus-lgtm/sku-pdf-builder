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

SLEEP_SECONDS = 0.15

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
          .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
            max-width: 980px;
          }

          /* Reduce vertical whitespace between components */
          div[data-testid="stVerticalBlock"] { gap: 0.6rem; }

          h1 { letter-spacing: -0.02em; margin-bottom: 0.15rem; }
          .subtitle { color: rgba(255,255,255,0.72); font-size: 0.95rem; margin-top: 0; }

          .card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            padding: 12px 14px;
            margin: 0.35rem 0;
          }

          .muted { color: rgba(255,255,255,0.70); font-size: 0.92rem; }

          .successbox {
            background: rgba(0, 255, 140, 0.10);
            border: 1px solid rgba(0, 255, 140, 0.25);
            border-radius: 12px;
            padding: 10px 12px;
            margin-top: 0.35rem;
          }

          .warningbox {
            background: rgba(255, 200, 0, 0.10);
            border: 1px solid rgba(255, 200, 0, 0.25);
            border-radius: 12px;
            padding: 10px 12px;
            margin-top: 0.35rem;
          }

          .footer {
            margin-top: 12px;
            color: rgba(255,255,255,0.55);
            font-size: 0.85rem;
            text-align: center;
          }

          /* Make uploader look tighter */
          div[data-testid="stFileUploader"] section {
            border: 1px dashed rgba(255,255,255,0.25);
            border-radius: 12px;
            padding: 4px 6px 8px 6px;
            background: rgba(255,255,255,0.03);
          }
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

st.markdown("<h1>SKU Visual Analyzer</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an XLSX to generate a clickable PDF grid of product images.</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="muted"><b>Required:</b> SKU, Revenue &nbsp;&nbsp; <b>Optional:</b> Units, AUR</div>', unsafe_allow_html=True)
    st.download_button(
        label="Download XLSX template",
        data=build_template_bytes(),
        file_name="SKU_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.markdown("</div>", unsafe_allow_html=True)

brand_label = st.selectbox(
    "Brand",
    list(BRAND_OPTIONS.keys()),
    index=0,
    format_func=lambda x: f"{brand_icon(x)}  {x}",
)
brand_domain = BRAND_OPTIONS[brand_label]

uploaded_file = st.file_uploader("Upload XLSX", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        items_all = build_items_from_df(df, brand_domain)

        per_page = GRID_ROWS * GRID_COLS
        total_pages = (len(items_all) + per_page - 1) // per_page

        st.markdown(
            f'<div class="successbox"><b>Loaded:</b> {len(items_all)} unique SKUs &nbsp;&nbsp; <b>Pages:</b> {total_pages}</div>',
            unsafe_allow_html=True,
        )

        if st.button("Generate PDF", type="primary"):
            progress = st.progress(0)
            status = st.empty()

            pages_tiles = []
            processed = 0

            for page_num, chunk in enumerate(chunk_list(items_all, per_page), start=1):
                status.markdown(f"**Building page {page_num} of {total_pages}...**")
                page_tiles = []

                for item in chunk:
                    tile_img, link_url = make_tile_image(item)
                    page_tiles.append((tile_img, link_url))

                    processed += 1
                    progress.progress(min(1.0, processed / len(items_all)))

                pages_tiles.append(page_tiles)

            status.markdown("**Building PDF...**")
            pdf_buffer = build_clickable_pdf(pages_tiles)

            file_name = f"SKU_Visual_Analyzer_{brand_domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            st.markdown('<div class="successbox"><b>Done.</b> Your PDF is ready.</div>', unsafe_allow_html=True)

            st.download_button(
                label="Download PDF",
                data=pdf_buffer,
                file_name=file_name,
                mime="application/pdf",
            )

    except Exception as e:
        st.markdown(f'<div class="warningbox"><b>Error:</b> {str(e)}</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Each PDF tile is clickable and opens the SKU product page.</div>', unsafe_allow_html=True)

