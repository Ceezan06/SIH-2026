"""
merge_into_projects.py  (v2 - supports both CSV and live-API sources)

This version understands TWO different "dialects" of field names:
  - CSV dialect (from load_raw_csvs.py): "Hon'ble Members of Parliament",
    "Recommended date", "Completion Date", etc.
  - Live API dialect (from automated_ingestion.py): "MP_NAME",
    "RECOMMENDATION_DATE", "ACTUAL_END_DATE", etc.

get_field() now accepts EITHER a single concept name OR a list of
alternative concept names, and returns the first one it finds - so the
same merge logic works no matter which pipeline the row came from.

Everything else works exactly like before.
"""

import math
import re

import pandas as pd
import psycopg2


# ============================================================
# CONFIG
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "Arja200507$$",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_key(k: str) -> str:
    k = k.replace("'", "")
    k = re.sub(r"\(.*?\)", "", k)
    k = re.sub(r"[^a-zA-Z ]", " ", k)
    k = re.sub(r"\s+", " ", k).strip().lower()
    return k


def get_field(payload: dict, concepts):
    """
    Look up a value by its general meaning, not exact spelling.
    'concepts' can be a single string or a list of alternative names to try,
    e.g. get_field(row, ["recommended date", "recommendation date"]).
    Returns the first match found, or None.
    """
    if isinstance(concepts, str):
        concepts = [concepts]

    for concept in concepts:
        target = normalize_key(concept)
        for k, v in payload.items():
            if normalize_key(k) == target:
                if v is None:
                    continue
                if isinstance(v, float) and math.isnan(v):
                    continue
                if str(v).strip() == "" or str(v).strip().upper() == "NA":
                    continue
                return v
    return None


def parse_district(ida_value):
    if not ida_value:
        return None
    name = str(ida_value).split("(")[0].strip()
    return name or None


def parse_date(value):
    if not value:
        return None
    parsed = pd.to_datetime(str(value), dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def parse_amount(value):
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def load_payloads_json(payload):
    if isinstance(payload, str):
        import json
        return json.loads(payload)
    return payload


def resolve_work_id(rec, sanc, comp, exp, fallback_id):
    """
    Picks the best available work identifier for this group of records.

    Live-API rows all share a WORK_RECOMMENDATION_DTL_ID field - this is
    the most reliable key when present, since it's consistent across all
    four stages for API-sourced data.

    CSV rows don't have this field, so we fall back to the work_id that
    was already extracted at ingestion time (stored as source_record_id
    in raw_landing, passed in here as 'fallback_id').
    """
    for source in (sanc, rec, comp, exp):
        dtl_id = source.get("WORK_RECOMMENDATION_DTL_ID")
        if dtl_id:
            return f"DTL-{dtl_id}"   # prefixed so it never collides with a CSV-style work_id
    return fallback_id


# ============================================================
# DATABASE HELPERS - unchanged from before
# ============================================================

def get_or_create_mp(cursor, cache, name, constituency, state):
    if not name:
        return None
    key = (name.strip().upper(), (constituency or "").strip().upper())
    if key in cache:
        return cache[key]
    cursor.execute(
        """
        INSERT INTO mps (name, constituency, state)
        VALUES (%s, %s, %s)
        ON CONFLICT (name, constituency) DO UPDATE SET state = EXCLUDED.state
        RETURNING mp_id
        """,
        (name.strip(), constituency.strip() if constituency else None, state),
    )
    mp_id = cursor.fetchone()[0]
    cache[key] = mp_id
    return mp_id


def get_or_create_district(cursor, cache, district_name, state):
    if not district_name:
        return None
    key = (district_name.strip().upper(), (state or "").strip().upper())
    if key in cache:
        return cache[key]
    cursor.execute(
        """
        INSERT INTO districts (district_name, state)
        VALUES (%s, %s)
        ON CONFLICT (district_name, state) DO UPDATE SET state = EXCLUDED.state
        RETURNING district_id
        """,
        (district_name.strip(), state),
    )
    district_id = cursor.fetchone()[0]
    cache[key] = district_id
    return district_id


def get_or_create_vendor(cursor, cache, vendor_name):
    if not vendor_name:
        return None
    key = vendor_name.strip().upper()
    if key in cache:
        return cache[key]
    cursor.execute(
        """
        INSERT INTO vendors (vendor_name)
        VALUES (%s)
        ON CONFLICT (vendor_name) DO NOTHING
        RETURNING vendor_id
        """,
        (vendor_name.strip(),),
    )
    row = cursor.fetchone()
    if row is None:
        cursor.execute("SELECT vendor_id FROM vendors WHERE vendor_name = %s", (vendor_name.strip(),))
        row = cursor.fetchone()
    vendor_id = row[0]
    cache[key] = vendor_id
    return vendor_id


def fetch_source_rows(cursor, source_label):
    """Returns {work_id: payload_dict} for one source. work_id here is the
    RAW extracted id from ingestion time - resolve_work_id() may override it later."""
    cursor.execute(
        "SELECT source_record_id, payload FROM raw_landing WHERE source = %s",
        (source_label,),
    )
    result = {}
    for work_id, payload in cursor.fetchall():
        if not work_id:
            continue
        result[work_id] = load_payloads_json(payload)
    return result


# ============================================================
# MAIN LOGIC
# ============================================================

def main():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    print("Reading raw_landing rows for each lifecycle stage...")
    recommended = fetch_source_rows(cur, "esakshi_recommended")
    sanctioned = fetch_source_rows(cur, "esakshi_sanctioned")
    completed = fetch_source_rows(cur, "esakshi_completed")
    expenditure = fetch_source_rows(cur, "esakshi_expenditure")

    # Group everything by its RESOLVED work_id (preferring WORK_RECOMMENDATION_DTL_ID
    # when present) rather than the raw ingestion-time id, so CSV and API records
    # for the truly same work end up merged into one row wherever possible.
    #
    # IMPORTANT: each group slot is created (via setdefault) at the exact moment
    # it's needed, not in a separate pre-pass. An earlier version pre-built all
    # slots from a merged dict first, which could silently skip creating a slot
    # if a raw id happened to collide across sources - causing a KeyError later.
    # Building slots on-demand like this makes that failure mode impossible.
    def add_to_group(grouped_dict, stage_key, source_dict):
        for raw_id, payload in source_dict.items():
            dtl_id = payload.get("WORK_RECOMMENDATION_DTL_ID")
            resolved_id = f"DTL-{dtl_id}" if dtl_id else raw_id
            grouped_dict.setdefault(resolved_id, {"rec": {}, "sanc": {}, "comp": {}, "exp": {}})
            grouped_dict[resolved_id][stage_key] = payload

    grouped = {}
    add_to_group(grouped, "rec", recommended)
    add_to_group(grouped, "sanc", sanctioned)
    add_to_group(grouped, "comp", completed)
    add_to_group(grouped, "exp", expenditure)

    print(f"Found {len(grouped)} unique works to merge.\n")

    mp_cache, district_cache, vendor_cache = {}, {}, {}
    written = 0

    try:
        for work_id, stages in sorted(grouped.items()):
            rec, sanc, comp, exp = stages["rec"], stages["sanc"], stages["comp"], stages["exp"]

            mp_name = get_field(sanc, ["honble members of parliament", "mp name"]) \
                or get_field(rec, ["honble members of parliament", "mp name"]) \
                or get_field(comp, ["honble members of parliament", "mp name"]) \
                or get_field(exp, ["honble members of parliament", "mp name"])

            constituency = get_field(sanc, "constituency") or get_field(rec, "constituency") \
                or get_field(comp, "constituency") or get_field(exp, "constituency")

            state = get_field(sanc, ["state", "state name"]) or get_field(rec, ["state", "state name"]) \
                or get_field(comp, ["state", "state name"]) or get_field(exp, ["state", "state name"])

            ida = get_field(sanc, ["ida", "ida name"]) or get_field(rec, ["ida", "ida name"]) \
                or get_field(comp, ["ida", "ida name"]) or get_field(exp, ["ida", "ida name"])
            district_name = parse_district(ida)

            category = get_field(rec, "work category") or get_field(sanc, "work category") \
                or get_field(comp, "work category")

            description = get_field(sanc, "work description") or get_field(rec, "work description") \
                or get_field(comp, "work description") or get_field(exp, ["work description", "activity name"])

            recommended_amount = parse_amount(get_field(rec, "recommended amount"))
            recommended_date = parse_date(get_field(rec, ["recommended date", "recommendation date"]))

            sanctioned_amount = parse_amount(get_field(sanc, "sanction amount"))
            sanction_date = parse_date(
                get_field(sanc, "sanction date") or get_field(rec, "sanction date")
            )
            work_status = get_field(sanc, ["work status", "work stage"])

            actual_completion_date = parse_date(get_field(comp, ["completion date", "actual end date"]))
            amount_disbursed = parse_amount(get_field(comp, ["amount disbursed", "actual amount"]))

            vendor_name = get_field(exp, "vendor name")
            payment_status = get_field(exp, ["payment status", "work status"])
            expenditure_date = parse_date(get_field(exp, "expenditure date"))

            mp_id = get_or_create_mp(cur, mp_cache, mp_name, constituency, state)
            district_id = get_or_create_district(cur, district_cache, district_name, state)
            vendor_id = get_or_create_vendor(cur, vendor_cache, vendor_name)

            cur.execute(
                """
                INSERT INTO projects (
                    work_id, mp_id, district_id, vendor_id,
                    category, description,
                    recommended_amount, recommended_date,
                    sanctioned_amount, sanction_date, work_status,
                    actual_completion_date, amount_disbursed,
                    payment_status, expenditure_date
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (work_id) DO UPDATE SET
                    mp_id = EXCLUDED.mp_id,
                    district_id = EXCLUDED.district_id,
                    vendor_id = COALESCE(EXCLUDED.vendor_id, projects.vendor_id),
                    category = EXCLUDED.category,
                    description = EXCLUDED.description,
                    recommended_amount = EXCLUDED.recommended_amount,
                    recommended_date = EXCLUDED.recommended_date,
                    sanctioned_amount = EXCLUDED.sanctioned_amount,
                    sanction_date = EXCLUDED.sanction_date,
                    work_status = EXCLUDED.work_status,
                    actual_completion_date = COALESCE(EXCLUDED.actual_completion_date, projects.actual_completion_date),
                    amount_disbursed = COALESCE(EXCLUDED.amount_disbursed, projects.amount_disbursed),
                    payment_status = COALESCE(EXCLUDED.payment_status, projects.payment_status),
                    expenditure_date = COALESCE(EXCLUDED.expenditure_date, projects.expenditure_date)
                """,
                (
                    work_id, mp_id, district_id, vendor_id,
                    category, description,
                    recommended_amount, recommended_date,
                    sanctioned_amount, sanction_date, work_status,
                    actual_completion_date, amount_disbursed,
                    payment_status, expenditure_date,
                ),
            )
            written += 1

        conn.commit()
        print(f"Done. {written} project rows inserted/updated.")
        print(f"MPs touched: {len(mp_cache)} | Districts touched: {len(district_cache)} | Vendors touched: {len(vendor_cache)}")

    except Exception as e:
        conn.rollback()
        print("\nSomething went wrong - no changes were saved. Error details:")
        print(e)

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
