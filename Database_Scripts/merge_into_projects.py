"""
merge_into_projects.py

What this script does, in plain terms:
  1. Reads all the rows out of raw_landing (the untouched CSV data).
  2. Groups them by which lifecycle stage they came from: Recommended,
     Sanctioned, Completed, Expenditure.
  3. For every unique work_id, it stitches together whatever info exists
     across those four stages into ONE combined record.
  4. It looks up (or creates) the matching MP, District, and Vendor rows.
  5. It inserts (or updates, if run again later) one row per work_id into
     the projects table.

You should NOT need to edit any logic below - only the CONFIG section
at the top needs your own details filled in (same as the last script).

Safe to re-run: if you load new CSVs into raw_landing later and run this
again, it will UPDATE existing projects with new info (e.g. a work that
was only "Recommended" before now has a "Completed" stage) rather than
creating duplicates.
"""

import math
import re

import pandas as pd
import psycopg2


# ============================================================
# CONFIG - edit this to match your setup (same as load_raw_csvs.py)
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "Arja200507$$",   # <-- put your Postgres password here
}


# ============================================================
# HELPER FUNCTIONS - you shouldn't need to touch these
# ============================================================

def normalize_key(k: str) -> str:
    """
    Turns a messy column header into a simple comparable form, e.g.:
      "Hon'ble Members of Parliament"  -> "honble members of parliament"
      "Sanction Amount ( \u20b9 )"          -> "sanction amount"
      "Work Category"                  -> "work category"
    This lets us match the same underlying field across CSVs that spell
    their headers slightly differently.
    """
    k = k.replace("'", "")
    k = re.sub(r"\(.*?\)", "", k)
    k = re.sub(r"[^a-zA-Z ]", " ", k)
    k = re.sub(r"\s+", " ", k).strip().lower()
    return k


def get_field(payload: dict, concept: str):
    """Look up a value in a raw row's JSON by its general meaning, not exact spelling."""
    target = normalize_key(concept)
    for k, v in payload.items():
        if normalize_key(k) == target:
            if v is None:
                return None
            if isinstance(v, float) and math.isnan(v):
                return None
            if str(v).strip() == "":
                return None
            return v
    return None


def parse_district(ida_value):
    """'South 24 Parganas(DISTRICT MAGISTRATE ...)' -> 'South 24 Parganas'"""
    if not ida_value:
        return None
    name = str(ida_value).split("(")[0].strip()
    return name or None


def parse_date(value):
    """Turns '11-Feb-2025' style strings into a real date, or None if missing/invalid."""
    if not value:
        return None
    parsed = pd.to_datetime(str(value), dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def parse_amount(value):
    """Turns '34,259' or 34259 into a clean number, or None if missing/invalid."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def load_payloads_json(payload):
    """psycopg2 usually hands back jsonb as a dict already; handle the string case too."""
    if isinstance(payload, str):
        import json
        return json.loads(payload)
    return payload


# ============================================================
# DATABASE HELPERS - "get or create" pattern for reference tables
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
        # already existed, conflict skipped the insert - look it up instead
        cursor.execute("SELECT vendor_id FROM vendors WHERE vendor_name = %s", (vendor_name.strip(),))
        row = cursor.fetchone()
    vendor_id = row[0]
    cache[key] = vendor_id
    return vendor_id


def fetch_source_rows(cursor, source_label):
    """Returns {work_id: payload_dict} for one source, e.g. 'esakshi_recommended'."""
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

    all_work_ids = set(recommended) | set(sanctioned) | set(completed) | set(expenditure)
    print(f"Found {len(all_work_ids)} unique work_ids to merge.\n")

    mp_cache, district_cache, vendor_cache = {}, {}, {}
    written = 0

    try:
        for work_id in sorted(all_work_ids):
            rec = recommended.get(work_id, {})
            sanc = sanctioned.get(work_id, {})
            comp = completed.get(work_id, {})
            exp = expenditure.get(work_id, {})

            # Prefer the latest lifecycle stage available for shared fields,
            # since later stages are generally more authoritative.
            mp_name = (get_field(sanc, "honble members of parliament")
                       or get_field(rec, "honble members of parliament")
                       or get_field(comp, "honble members of parliament")
                       or get_field(exp, "honble members of parliament"))
            constituency = (get_field(sanc, "constituency") or get_field(rec, "constituency")
                            or get_field(comp, "constituency") or get_field(exp, "constituency"))
            state = (get_field(sanc, "state") or get_field(rec, "state")
                     or get_field(comp, "state") or get_field(exp, "state"))
            ida = (get_field(sanc, "ida") or get_field(rec, "ida")
                   or get_field(comp, "ida") or get_field(exp, "ida"))
            district_name = parse_district(ida)

            category = (get_field(rec, "work category") or get_field(sanc, "work category")
                        or get_field(comp, "work category"))
            description = (get_field(sanc, "work description") or get_field(rec, "work description")
                            or get_field(comp, "work description"))

            recommended_amount = parse_amount(get_field(rec, "recommended amount"))
            recommended_date = parse_date(get_field(rec, "recommended date"))

            sanctioned_amount = parse_amount(get_field(sanc, "sanction amount"))
            sanction_date = parse_date(get_field(sanc, "sanction date") or get_field(rec, "sanction date"))
            work_status = get_field(sanc, "work status")

            actual_completion_date = parse_date(get_field(comp, "completion date"))
            amount_disbursed = parse_amount(get_field(comp, "amount disbursed"))

            vendor_name = get_field(exp, "vendor name")
            payment_status = get_field(exp, "payment status")
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
