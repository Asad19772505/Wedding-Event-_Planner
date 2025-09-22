# app.py
import io
import math
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Wedding Per-Guest Cost (Multan, PK)", layout="wide")
st.title("💍 Wedding Per-Guest Cost — Multan, Pakistan")

st.caption(
    "Upload your expenses CSV or use the sample. Adjust assumptions (guests, per-plate, taxes, "
    "service charges, wastage, contingency, discounts) and compare up to three scenarios."
)

# -----------------------------
# Helpers & Defaults
# -----------------------------
TEMPLATE_COLUMNS = [
    "Category",          # e.g., "Venue", "Catering", "Decor", "Photography"
    "Item",              # e.g., "Hall Rent", "Stage", "Lights"
    "CostType",          # "Fixed" or "PerGuest"
    "UnitCost",          # numeric; PKR
    "Qty",               # numeric; for Fixed = count; for PerGuest = qty per guest (often 1)
    "TaxRatePct",        # optional; if blank, treated as 0 (applies to this row only)
]

def sample_expenses() -> pd.DataFrame:
    data = [
        # Fixed costs
        ["Venue", "Hall Rent (evening)", "Fixed", 250000, 1, 0],
        ["Venue", "Generator/Backup",    "Fixed", 35000,  1, 0],
        ["Decor", "Stage + Backdrop",    "Fixed", 120000, 1, 0],
        ["Decor", "Lighting Package",    "Fixed", 45000,  1, 0],
        ["Media", "Photo + Video",       "Fixed", 90000,  1, 0],
        ["Logistics", "Transport/Valet", "Fixed", 30000,  1, 0],
        # Per-guest add-ons (beyond catering/plate)
        ["Hospitality", "Welcome Drinks", "PerGuest", 150, 1, 0],
        ["Gifting", "Mehndi/Token",       "PerGuest", 120, 1, 0],
    ]
    return pd.DataFrame(data, columns=TEMPLATE_COLUMNS)

def make_template_csv() -> bytes:
    df = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    return df.to_csv(index=False).encode("utf-8")

def coerce_numeric(series, default=0.0):
    """Safely coerce to numeric; fill NaNs with default."""
    out = pd.to_numeric(series, errors="coerce").fillna(default)
    return out

def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Validate required columns; coerce numerics; return (clean_df, warnings)."""
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
    bad_types = ~df["CostType"].str.replace(" ", "").isin(valid_types)
    if bad_types.any():
        bad_vals = sorted(df.loc[bad_types, "CostType"].unique().tolist())
        raise ValueError(
            f"Invalid CostType values found: {bad_vals}. Use only 'Fixed' or 'PerGuest'."
        )

    # Standardize to "PerGuest"
    df.loc[df["CostType"].str.replace(" ", "") == "Perguest", "CostType"] = "PerGuest"

    # Coerce numerics
    df["UnitCost"] = coerce_numeric(df["UnitCost"], default=0.0)
    df["Qty"] = coerce_numeric(df["Qty"], default=1.0)
    if (df["Qty"] <= 0).any():
        warnings.append("Some Qty values were ≤ 0; they will still be used as-is.")

    if "TaxRatePct" in df.columns:
        df["TaxRatePct"] = coerce_numeric(df["TaxRatePct"], default=0.0)
    else:
        df["TaxRatePct"] = 0.0

    return df, warnings

def money(x: float) -> str:
    try:
        return f"PKR {x:,.0f}"
    except Exception:
        return f"PKR {x}"

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
    """
    Compute total & per-guest cost with detailed components.
    - Per-plate cost is treated separately (catering).
    - Wastage applies to catering only.
    - Row-level TaxRatePct applies to that row, then global service/tax apply on subtotal.
    - Contingency applies to (subtotal + service + tax) before discounts.
    - Discounts are applied at the end (total flat + per-guest).
    """
    guests = max(int(guests), 0)
    per_plate = max(float(per_plate), 0.0)
    wastage = max(float(wastage_pct), 0.0) / 100.0
    svc = max(float(global_service_pct), 0.0) / 100.0
    tax = max(float(global_tax_pct), 0.0) / 100.0
    contingency = max(float(contingency_pct), 0.0) / 100.0
    discount_total = max(float(discount_total), 0.0)
    discount_per_guest = max(float(discount_per_guest), 0.0)

    # Catering (plate) with wastage
    catered_guests = math.ceil(guests * (1.0 + wastage))
    catering_cost = catered_guests * per_plate

    # Fixed vs Per-Guest rows from uploaded CSV
    fixed_rows = df_costs[df_costs["CostType"] == "Fixed"].copy()
    per_guest_rows = df_costs[df_costs["CostType"] == "PerGuest"].copy()

    # Apply row-level tax only to those rows (common in Pakistan: some items taxed differently)
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

    # Global service & tax after row-level taxes
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

def breakdown_table(break: Dict[str, float]) -> pd.DataFrame:
    rows = [
        ("Guests (input)", break["Guests_Input"]),
        ("Catering plates charged (incl. wastage)", break["Catering_Plates_Charged"]),
        ("Catering cost", break["Catering_Cost"]),
        ("Fixed subtotal", break["Fixed_Subtotal"]),
        ("Per-guest subtotal (extras)", break["PerGuest_Subtotal"]),
        ("Row-level taxes (per line)", break["Row_Level_Taxes"]),
        ("Global service charge", break["Global_Service"]),
        ("Global tax", break["Global_Tax"]),
        ("Contingency", break["Contingency"]),
        ("Total before discounts", break["Total_Before_Discounts"]),
        ("Discounts (flat + per-guest)", -break["Discounts"]),
        ("Grand total", break["Grand_Total"]),
        ("Per-guest cost", break["Per_Guest_Cost"]),
    ]
    df = pd.DataFrame(rows, columns=["Component", "Amount (PKR)"])
    return df

# -----------------------------
# Sidebar — Global Inputs
# -----------------------------
with st.sidebar:
    st.header("⚙️ Global Assumptions")

    st.subheader("Catering (per plate, PKR)")
    base_guests = st.number_input("Guests (Scenario 1 default)", min_value=0, value=300, step=10)
    plate_cost = st.number_input("Per-plate cost (PKR)", min_value=0, value=1900, step=50)
    wastage_pct = st.slider("Wastage % on catering", 0, 20, 5, help="Adds extra plates on top of guest count.")

    st.subheader("Surcharges & Safety")
    service_pct = st.number_input("Global service charge %", min_value=0.0, value=2.0, step=0.5)
    tax_pct = st.number_input("Global tax %", min_value=0.0, value=0.0, step=0.5,
                               help="Set to 0 if your per-line TaxRatePct already covers taxes.")
    contingency_pct = st.number_input("Contingency %", min_value=0.0, value=5.0, step=0.5)

    st.subheader("Discounts")
    discount_total = st.number_input("Flat discount (PKR)", min_value=0.0, value=0.0, step=1000.0)
    discount_per_guest = st.number_input("Per-guest discount (PKR)", min_value=0.0, value=0.0, step=10.0)

# -----------------------------
# File Upload & Template
# -----------------------------
with st.expander("📄 Upload Expenses CSV (or use sample)"):
    st.write(
        "CSV must include columns (case-sensitive): "
        f"`{', '.join(TEMPLATE_COLUMNS)}`.\n\n"
        "- **CostType** = `Fixed` or `PerGuest`\n"
        "- **UnitCost** in PKR\n"
        "- **Qty** for Fixed = count; for PerGuest = quantity *per guest* (often 1)\n"
        "- **TaxRatePct** optional (row-level tax)"
    )
    col_dl, col_up = st.columns([1, 2])
    with col_dl:
        st.download_button(
            "Download empty template CSV",
            data=make_template_csv(),
            file_name="wedding_expenses_template.csv",
            mime="text/csv",
        )
    with col_up:
        uploaded = st.file_uploader("Upload your expenses CSV", type=["csv"])

    if uploaded is None:
        st.info("No file uploaded — using sample expenses. You can upload anytime.")
        df_exp = sample_expenses()
    else:
        try:
            df_exp = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

    try:
        df_exp, _warn = validate_and_clean(df_exp)
        if _warn:
            for w in _warn:
                st.warning(w)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.dataframe(df_exp, use_container_width=True)

# -----------------------------
# Scenarios (3 tabs)
# -----------------------------
st.subheader("📊 Scenario Comparison")

tabs = st.tabs(["Scenario 1", "Scenario 2", "Scenario 3"])

def scenario_controls(default_guests, default_plate, key_prefix: str):
    c1, c2 = st.columns(2)
    with c1:
        guests = st.number_input("Guests", min_value=0, value=default_guests, step=10, key=f"{key_prefix}_guests")
    with c2:
        plate = st.number_input("Per-plate (PKR)", min_value=0, value=default_plate, step=50, key=f"{key_prefix}_plate")
    return guests, plate

results = []
for i, tab in enumerate(tabs, start=1):
    with tab:
        st.markdown(f"**Configure Scenario {i}**")
        guests_i, plate_i = scenario_controls(base_guests, plate_cost, key_prefix=f"s{i}")

        # Compute
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

        # Display
        st.metric("Per-guest cost", money(b["Per_Guest_Cost"]))
        st.metric("Grand total", money(b["Grand_Total"]))

        st.markdown("**Detailed Breakdown**")
        st.dataframe(
            breakdown_table(b).assign(**{"Amount (PKR)": lambda d: d["Amount (PKR)"].map(lambda x: f"{x:,.0f}")}),
            use_container_width=True
        )

# -----------------------------
# Comparison Table & Sensitivities
# -----------------------------
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

# Download results
csv_buf = io.StringIO()
summary_df.to_csv(csv_buf, index=False)
st.download_button("⬇️ Download Scenario Summary (CSV)", data=csv_buf.getvalue(), file_name="wedding_scenarios_summary.csv", mime="text/csv")

# -----------------------------
# Quick Sensitivity (optional)
# -----------------------------
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
        st.write("Sensitivity grid will show **per-guest cost**.")

    # Build grid carefully
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
