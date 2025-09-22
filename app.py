# app.py
import io
import math
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# Page setup
# =============================================================================
st.set_page_config(page_title="Wedding Per-Guest Cost — Multan, Pakistan", layout="wide")
st.title("💍 Wedding Per-Guest Cost — Multan, Pakistan")

st.caption(
    "Upload a CSV of expenses OR build one from Multan vendor presets below. "
    "Adjust assumptions (guests, per-plate, taxes, service, wastage, contingency, discounts) "
    "and compare up to three scenarios. All currency PKR. "
    "Note: Provincial sales tax (PRA) rates and rules can change — treat presets as editable guidance, not tax advice."
)

# =============================================================================
# Template / Schema
# =============================================================================
TEMPLATE_COLUMNS = [
    "Category",          # e.g., "Venue", "Catering", "Decor", "Photography"
    "Item",              # e.g., "Hall Rent", "Stage", "Lights"
    "CostType",          # "Fixed" or "PerGuest"
    "UnitCost",          # numeric; PKR
    "Qty",               # numeric; for Fixed = count; for PerGuest = qty per guest (often 1)
    "TaxRatePct",        # optional; if blank, treated as 0 (applies to this row only)
]

def make_empty_template_csv() -> bytes:
    df = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    return df.to_csv(index=False).encode("utf-8")

def sample_expenses() -> pd.DataFrame:
    data = [
        ["Venue", "Hall Rent (evening)", "Fixed", 250000, 1, 0],
        ["Venue", "Generator/Backup",    "Fixed", 35000,  1, 0],
        ["Decor", "Stage + Backdrop",    "Fixed", 120000, 1, 0],
        ["Decor", "Lighting Package",    "Fixed", 45000,  1, 0],
        ["Media", "Photo + Video",       "Fixed", 90000,  1, 0],
        ["Logistics", "Transport/Valet", "Fixed", 30000,  1, 0],
        ["Hospitality", "Welcome Drinks", "PerGuest", 150, 1, 0],
        ["Gifting", "Mehndi/Token",       "PerGuest", 120, 1, 0],
    ]
    return pd.DataFrame(data, columns=TEMPLATE_COLUMNS)

def coerce_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)

def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings = []
    missing = [c for c in TEMPLATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Your file is missing required columns: {missing}. "
            f"Expected columns: {TEMPLATE_COLUMNS}"
        )

    df = df.copy()
    # Normalize CostType
    df["CostType"] = df["CostType"].astype(str).str.strip().str.title()
    valid_types = {"Fixed", "Perguest"}
    mask_bad = ~df["CostType"].str.replace(" ", "").isin(valid_types)
    if mask_bad.any():
        bad_vals = sorted(df.loc[mask_bad, "CostType"].unique().tolist())
        raise ValueError(
            f"Invalid CostType values found: {bad_vals}. Use only 'Fixed' or 'PerGuest'."
        )
    df.loc[df["CostType"].str.replace(" ", "") == "Perguest", "CostType"] = "PerGuest"

    # Numerics
    df["UnitCost"] = coerce_numeric(df["UnitCost"], default=0.0)
    df["Qty"] = coerce_numeric(df["Qty"], default=1.0)
    if (df["Qty"] <= 0).any():
        warnings.append("Some Qty values were ≤ 0; they will still be used as-is.")

    if "TaxRatePct" in df.columns:
        df["TaxRatePct"] = coerce_numeric(df["TaxRatePct"], default=0.0)
    else:
        df["TaxRatePct"] = 0.0

    # Clean strings
    for c in ["Category", "Item"]:
        df[c] = df[c].astype(str).str.strip()

    return df, warnings

def money(x: float) -> str:
    try:
        return f"PKR {x:,.0f}"
    except Exception:
        return f"PKR {x}"

# =============================================================================
# Multan Vendor Presets (editable)
# =============================================================================
# Editable Punjab (PRA) default tax assumptions by category (you can tweak in UI).
DEFAULT_CATEGORY_TAX = {
    "Catering": 16.0,      # PRA sales tax on catering services (editable)
    "Venue": 16.0,         # halls/marquees as services (editable)
    "Decor": 16.0,         # event decor services (editable)
    "Media": 0.0,          # photography/videography often exempt/varies; keep editable
    "Logistics": 0.0,      # transport/logistics (editable)
    "Hospitality": 0.0,    # welcome drinks if handled outside caterer (editable)
    "Gifting": 0.0,        # favors (often goods; treatment varies) — editable
}

CATERING_TIERS = {
    "Economy — PKR 1,500": 1500,
    "Standard — PKR 1,900": 1900,
    "Premium — PKR 2,500": 2500,
}

VENUES = {
    "City Marquee (Multan)": 220000,
    "Royal Orchard Club (Multan)": 320000,
    "Premier Event Hall (Multan)": 400000,
}

DECOR_PACKS = {
    "Basic Decor (Stage + Backdrop + Lights)": 90000,
    "Elegant Decor (Stage + Floral + Lights)": 150000,
    "Grand Decor (Stage + Floral + Ceiling Drapes + Lights)": 220000,
}

MEDIA_PACKS = {
    "Photography (Basic)": 60000,
    "Photo + Video (Standard)": 100000,
    "Cinematic Package (Photo + 2-Cam Video + Drone)": 160000,
}

LOGISTICS_PACKS = {
    "Guest Transport/Valet": 30000,
    "Generator/Backup Power": 40000,
}

HOSPITALITY_PER_GUEST = {
    "Welcome Drinks": 150,
    "Live Tea Station": 120,
}

GIFTING_PER_GUEST = {
    "Favors/Mehndi Token": 120,
    "Custom Gift Box": 300,
}

def make_df_from_presets(
    venue_name: str,
    decor_name: str,
    media_name: str,
    logistics_selected: List[str],
    hospitality_selected: List[str],
    gifting_selected: List[str],
    category_tax_map: Dict[str, float],
) -> pd.DataFrame:
    rows = []

    # Venue
    if venue_name:
        rows.append(["Venue", venue_name, "Fixed", VENUES[venue_name], 1, category_tax_map.get("Venue", 0.0)])

    # Decor
    if decor_name:
        rows.append(["Decor", decor_name, "Fixed", DECOR_PACKS[decor_name], 1, category_tax_map.get("Decor", 0.0)])

    # Media
    if media_name:
        rows.append(["Media", media_name, "Fixed", MEDIA_PACKS[media_name], 1, category_tax_map.get("Media", 0.0)])

    # Logistics
    for lg in logistics_selected:
        rows.append(["Logistics", lg, "Fixed", LOGISTICS_PACKS[lg], 1, category_tax_map.get("Logistics", 0.0)])

    # Hospitality (PerGuest)
    for h in hospitality_selected:
        rows.append(["Hospitality", h, "PerGuest", HOSPITALITY_PER_GUEST[h], 1, category_tax_map.get("Hospitality", 0.0)])

    # Gifting (PerGuest)
    for g in gifting_selected:
        rows.append(["Gifting", g, "PerGuest", GIFTING_PER_GUEST[g], 1, category_tax_map.get("Gifting", 0.0)])

    return pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)

# =============================================================================
# Cost engine
# =============================================================================
def compute_breakdown(
    df_costs: pd.DataFrame,
    guests: int,
    per_plate: float,
    wastage_pct: float,
    global_service_pct: float,
    global_tax_pct: float,
    contingency_pct: float,
    discount_total: float,
    discount_per_guest: float,
) -> Dict[str, float]:
    guests = max(int(guests), 0)
    per_plate = max(float(per_plate), 0.0)
    wastage = max(float(wastage_pct), 0.0) / 100.0
    svc = max(float(global_service_pct), 0.0) / 100.0
    tax = max(float(global_tax_pct), 0.0) / 100.0
    contingency = max(float(contingency_pct), 0.0) / 100.0
    discount_total = max(float(discount_total), 0.0)
    discount_per_guest = max(float(discount_per_guest), 0.0)

    # Catering plates incl. wastage
    catered_guests = math.ceil(guests * (1.0 + wastage))
    catering_cost = catered_guests * per_plate

    # Fixed vs Per-Guest rows
    fixed_rows = df_costs[df_costs["CostType"] == "Fixed"].copy()
    per_guest_rows = df_costs[df_costs["CostType"] == "PerGuest"].copy()

    # Row-level taxes
    fixed_rows["RowSubtotal"] = fixed_rows["UnitCost"] * fixed_rows["Qty"]
    fixed_rows["RowTax"] = fixed_rows["RowSubtotal"] * (fixed_rows["TaxRatePct"] / 100.0)

    per_guest_rows["RowSubtotal"] = per_guest_rows["UnitCost"] * per_guest_rows["Qty"] * guests
    per_guest_rows["RowTax"] = per_guest_rows["RowSubtotal"] * (per_guest_rows["TaxRatePct"] / 100.0)

    fixed_subtotal = float(fixed_rows["RowSubtotal"].sum())
    fixed_row_tax = float(fixed_rows["RowTax"].sum())

    per_guest_subtotal = float(per_guest_rows["RowSubtotal"].sum())
    per_guest_row_tax = float(per_guest_rows["RowTax"].sum())

    subtotal_before_global = catering_cost + fixed_subtotal + per_guest_subtotal
    row_taxes = fixed_row_tax + per_guest_row_tax

    # Global service & tax
    base_for_service = subtotal_before_global + row_taxes
    service_charge = base_for_service * svc
    tax_charge = (base_for_service + service_charge) * tax

    # Contingency
    pre_cont_total = base_for_service + service_charge + tax_charge
    contingency_amt = pre_cont_total * contingency

    # Discounts
    total_before_discounts = pre_cont_total + contingency_amt
    total_discount = discount_total + (discount_per_guest * guests)

    grand_total = max(total_before_discounts - total_discount, 0.0)
    per_guest_cost = grand_total / guests if guests > 0 else 0.0

    return {
        "Guests_Input": guests,
        "Catering_Plates_Charged": catered_guests,
        "Catering_Cost": catering_cost,
        "Fixed_Subtotal": fixed_subtotal,
        "PerGuest_Subtotal": per_guest_subtotal,
        "Row_Level_Taxes": row_taxes,
        "Global_Service": service_charge,
        "Global_Tax": tax_charge,
        "Contingency": contingency_amt,
        "Total_Before_Discounts": total_before_discounts,
        "Discounts": total_discount,
        "Grand_Total": grand_total,
        "Per_Guest_Cost": per_guest_cost,
    }

def breakdown_table(b: Dict[str, float]) -> pd.DataFrame:
    rows = [
        ("Guests (input)", b["Guests_Input"]),
        ("Catering plates charged (incl. wastage)", b["Catering_Plates_Charged"]),
        ("Catering cost", b["Catering_Cost"]),
        ("Fixed subtotal", b["Fixed_Subtotal"]),
        ("Per-guest subtotal (extras)", b["PerGuest_Subtotal"]),
        ("Row-level taxes (per line)", b["Row_Level_Taxes"]),
        ("Global service charge", b["Global_Service"]),
        ("Global tax", b["Global_Tax"]),
        ("Contingency", b["Contingency"]),
        ("Total before discounts", b["Total_Before_Discounts"]),
        ("Discounts (flat + per-guest)", -b["Discounts"]),
        ("Grand total", b["Grand_Total"]),
        ("Per-guest cost", b["Per_Guest_Cost"]),
    ]
    return pd.DataFrame(rows, columns=["Component", "Amount (PKR)"])

# =============================================================================
# Sidebar — Global Assumptions
# =============================================================================
with st.sidebar:
    st.header("⚙️ Global Assumptions")

    st.subheader("Catering (Per Plate, PKR)")
    base_guests = st.number_input("Guests (Scenario 1 default)", min_value=0, value=300, step=10)
    # Choose a Multan catering tier
    catering_tier = st.selectbox("Catering Tier (Multan presets)", list(CATERING_TIERS.keys()), index=1)
    plate_cost = CATERING_TIERS[catering_tier]
    st.number_input("Per-plate (editable)", min_value=0, value=int(plate_cost), step=50, key="plate_edit", help="Override the tier price if needed.")
    plate_cost = st.session_state.get("plate_edit", plate_cost)
    wastage_pct = st.slider("Wastage % on catering", 0, 20, 5, help="Adds extra plates on top of guest count.")

    st.subheader("Surcharges & Safety")
    service_pct = st.number_input("Global service charge %", min_value=0.0, value=2.0, step=0.5)
    tax_pct = st.number_input("Global tax %", min_value=0.0, value=0.0, step=0.5,
                               help="Keep at 0 if per-line TaxRatePct already covers taxes.")
    contingency_pct = st.number_input("Contingency %", min_value=0.0, value=5.0, step=0.5)

    st.subheader("Discounts")
    discount_total = st.number_input("Flat discount (PKR)", min_value=0.0, value=0.0, step=1000.0)
    discount_per_guest = st.number_input("Per-guest discount (PKR)", min_value=0.0, value=0.0, step=10.0)

# =============================================================================
# Vendor Preset Builder (Multan)
# =============================================================================
st.subheader("🏗️ Build Expense Sheet from Multan Presets (optional)")
with st.expander("Click to expand Multan vendor catalog & tax presets"):
    c1, c2, c3 = st.columns(3)
    with c1:
        venue_choice = st.selectbox("Venue", ["(None)"] + list(VENUES.keys()), index=1)
        decor_choice = st.selectbox("Decor Package", ["(None)"] + list(DECOR_PACKS.keys()), index=2)
    with c2:
        media_choice = st.selectbox("Media Package", ["(None)"] + list(MEDIA_PACKS.keys()), index=2)
        logistics_multi = st.multiselect("Logistics (choose any)", list(LOGISTICS_PACKS.keys()),
                                         default=["Generator/Backup Power"])
    with c3:
        hospitality_multi = st.multiselect("Hospitality (per guest)", list(HOSPITALITY_PER_GUEST.keys()),
                                           default=["Welcome Drinks"])
        gifting_multi = st.multiselect("Gifting (per guest)", list(GIFTING_PER_GUEST.keys()),
                                       default=["Favors/Mehndi Token"])

    st.markdown("**Punjab (PRA) tax presets — edit per category**")
    tax_cols = st.columns(len(DEFAULT_CATEGORY_TAX))
    edited_tax = {}
    for i, (cat, default_rate) in enumerate(DEFAULT_CATEGORY_TAX.items()):
        with tax_cols[i]:
            edited_tax[cat] = st.number_input(f"{cat} tax %", min_value=0.0, max_value=30.0, value=float(default_rate), step=0.5)

    use_presets = st.checkbox("Use these presets to build my expense sheet", value=True,
                              help="If unchecked, the app will use your uploaded CSV or the sample data.")

# =============================================================================
# File Upload & Data Source
# =============================================================================
st.subheader("📄 Upload Expenses CSV (optional)")
col_dl, col_up = st.columns([1, 2])
with col_dl:
    st.download_button(
        "Download empty template CSV",
        data=make_empty_template_csv(),
        file_name="wedding_expenses_template.csv",
        mime="text/csv",
    )
with col_up:
    uploaded = st.file_uploader("Upload your expenses CSV", type=["csv"])

# Decide the expense DataFrame source (Presets > Uploaded > Sample)
if use_presets:
    df_exp = make_df_from_presets(
        venue_name=None if venue_choice == "(None)" else venue_choice,
        decor_name=None if decor_choice == "(None)" else decor_choice,
        media_name=None if media_choice == "(None)" else media_choice,
        logistics_selected=logistics_multi,
        hospitality_selected=hospitality_multi,
        gifting_selected=gifting_multi,
        category_tax_map=edited_tax,
    )
    if df_exp.empty:
        st.info("No presets selected yet — add items above or upload a CSV.")
        df_exp = sample_expenses()
else:
    if uploaded is not None:
        try:
            df_exp = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()
    else:
        st.info("No file uploaded — using sample expenses. You can upload anytime.")
        df_exp = sample_expenses()

# Validate & show
try:
    df_exp, _warn = validate_and_clean(df_exp)
    if _warn:
        for w in _warn:
            st.warning(w)
except Exception as e:
    st.error(str(e))
    st.stop()

st.dataframe(df_exp, use_container_width=True)

# =============================================================================
# Scenarios (3 tabs)
# =============================================================================
st.subheader("📊 Scenario Comparison")
tabs = st.tabs(["Scenario 1", "Scenario 2", "Scenario 3"])

def scenario_controls(default_guests, default_plate, key_prefix: str):
    c1, c2 = st.columns(2)
    with c1:
        guests = st.number_input("Guests", min_value=0, value=int(default_guests), step=10, key=f"{key_prefix}_guests")
    with c2:
        plate = st.number_input("Per-plate (PKR)", min_value=0, value=int(default_plate), step=50, key=f"{key_prefix}_plate")
    return guests, plate

results = []
for i, tab in enumerate(tabs, start=1):
    with tab:
        st.markdown(f"**Configure Scenario {i}**")
        guests_i, plate_i = scenario_controls(base_guests, plate_cost, key_prefix=f"s{i}")

        b = compute_breakdown(
            df_costs=df_exp,
            guests=guests_i,
            per_plate=plate_i,
            wastage_pct=wastage_pct,
            global_service_pct=service_pct,
            global_tax_pct=tax_pct,
            contingency_pct=contingency_pct,
            discount_total=discount_total,
            discount_per_guest=discount_per_guest,
        )
        results.append(b)

        cA, cB = st.columns(2)
        with cA:
            st.metric("Per-guest cost", money(b["Per_Guest_Cost"]))
        with cB:
            st.metric("Grand total", money(b["Grand_Total"]))

        st.markdown("**Detailed Breakdown**")
        st.dataframe(
            breakdown_table(b).assign(**{"Amount (PKR)": lambda d: d["Amount (PKR)"].map(lambda x: f"{x:,.0f}")}),
            use_container_width=True
        )

# =============================================================================
# Comparison Table & Download
# =============================================================================
st.divider()
st.subheader("📋 Scenario Summary")

summary_rows = []
for i, b in enumerate(results, start=1):
    summary_rows.append({
        "Scenario": f"Scenario {i}",
        "Guests": b["Guests_Input"],
        "Catering Plates (incl. wastage)": b["Catering_Plates_Charged"],
        "Grand Total (PKR)": round(b["Grand_Total"], 2),
        "Per-Guest Cost (PKR)": round(b["Per_Guest_Cost"], 2),
        "Fixed (PKR)": round(b["Fixed_Subtotal"], 2),
        "Per-Guest Extras (PKR)": round(b["PerGuest_Subtotal"], 2),
        "Catering (PKR)": round(b["Catering_Cost"], 2),
        "Service (PKR)": round(b["Global_Service"], 2),
        "Tax (PKR)": round(b["Global_Tax"], 2),
        "Contingency (PKR)": round(b["Contingency"], 2),
        "Discounts (PKR)": round(b["Discounts"], 2),
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(
    summary_df.style.format({
        "Grand Total (PKR)": "{:,.0f}",
        "Per-Guest Cost (PKR)": "{:,.0f}",
        "Fixed (PKR)": "{:,.0f}",
        "Per-Guest Extras (PKR)": "{:,.0f}",
        "Catering (PKR)": "{:,.0f}",
        "Service (PKR)": "{:,.0f}",
        "Tax (PKR)": "{:,.0f}",
        "Contingency (PKR)": "{:,.0f}",
        "Discounts (PKR)": "{:,.0f}",
    }),
    use_container_width=True
)

csv_buf = io.StringIO()
summary_df.to_csv(csv_buf, index=False)
st.download_button("⬇️ Download Scenario Summary (CSV)", data=csv_buf.getvalue(),
                   file_name="wedding_scenarios_summary.csv", mime="text/csv")

# =============================================================================
# Quick Sensitivity (optional)
# =============================================================================
st.divider()
with st.expander("📈 Quick Sensitivity (Guests × Per-plate)"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        g_min = st.number_input("Guests min", 50, 2000, max(0, base_guests - 100), step=10)
        g_max = st.number_input("Guests max", 50, 2000, base_guests + 200, step=10)
        g_step = st.number_input("Guests step", 10, 500, 50, step=10)
    with col_b:
        p_min = st.number_input("Per-plate min (PKR)", 200, 10000, max(0, plate_cost - 300), step=50)
        p_max = st.number_input("Per-plate max (PKR)", 200, 10000, plate_cost + 500, step=50)
        p_step = st.number_input("Per-plate step", 10, 2000, 100, step=10)
    with col_c:
        st.write("Sensitivity grid shows **per-guest cost** for the chosen ranges.")

    try:
        guests_grid = list(range(int(g_min), int(g_max) + 1, int(g_step)))
        plate_grid = list(range(int(p_min), int(p_max) + 1, int(p_step)))
        grid_records = []
        for g in guests_grid:
            for p in plate_grid:
                b = compute_breakdown(
                    df_costs=df_exp,
                    guests=g,
                    per_plate=p,
                    wastage_pct=wastage_pct,
                    global_service_pct=service_pct,
                    global_tax_pct=tax_pct,
                    contingency_pct=contingency_pct,
                    discount_total=discount_total,
                    discount_per_guest=discount_per_guest,
                )
                grid_records.append({"Guests": g, "PerPlate": p, "PerGuestCost": round(b["Per_Guest_Cost"], 2)})
        sens_df = pd.DataFrame(grid_records)
        pivot = sens_df.pivot(index="Guests", columns="PerPlate", values="PerGuestCost").sort_index()
        st.dataframe(pivot, use_container_width=True)
    except Exception as e:
        st.warning(f"Sensitivity not generated: {e}")
