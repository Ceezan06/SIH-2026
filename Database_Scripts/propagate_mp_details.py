"""
propagate_mp_details.py

Fills in dim_mp.party_name, dim_mp.membership_status, and
dim_mp.terms_served from staging.raw_sitting_members.

WHY MATCHING BY NAME WON'T WORK HERE:
Sitting_Members.xlsx stores names as "Surname, Title Firstname"
(e.g. "A, Shri Mani"), but dim_mp.mp_name comes from eSAKSHI in
"Title Firstname Surname" order (e.g. "Shri Mani A"). Matching those
directly would silently fail for almost every row.

Instead this matches on (constituency, state) - normalized to upper
case with SC/ST/etc. parenthetical suffixes stripped from both sides,
since casing and suffix formatting differ between the two sources
even though the underlying constituency is the same.

Lok Sabha only: this file has no Rajya Sabha data (RS members have no
constituency to match on anyway - dim_mp.constituency is NULL for
every RS row, so they're naturally skipped, not mismatched).

Run this AFTER load_csv_to_staging.py has populated
staging.raw_sitting_members.
"""

import re
import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "Arja200507$$",
}


def normalize_constituency(value):
    """'Nilgiris(SC)' / 'NILGIRIS (SC)' / 'nilgiris' -> 'NILGIRIS' """
    if not value:
        return None
    s = str(value).upper().strip()
    s = re.sub(r"\(.*?\)", "", s)      # drop (SC), (ST), (BR) etc.
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def normalize_state(value):
    if not value:
        return None
    return str(value).upper().strip()


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        cur.execute('''
            SELECT "Party Name", "Constituency", "State",
                   "MemberShip Status", "Lok Sabha Terms"
            FROM staging.raw_sitting_members
        ''')
        rows = cur.fetchall()
        print(f"Found {len(rows)} rows in staging.raw_sitting_members")

        matched, unmatched, ambiguous = 0, [], []

        for party, constituency, state, membership_status, terms in rows:
            norm_const = normalize_constituency(constituency)
            norm_state = normalize_state(state)

            if not norm_const or not norm_state:
                unmatched.append((constituency, state, "missing constituency or state"))
                continue

            cur.execute('''
                UPDATE dim_mp
                SET party_name = %s,
                    membership_status = %s,
                    terms_served = %s
                WHERE house = '2'
                  AND UPPER(TRIM(state)) = %s
                  AND REGEXP_REPLACE(UPPER(TRIM(constituency)), '\\(.*?\\)', '', 'g') = %s
                RETURNING mp_id
            ''', (party, membership_status, terms, norm_state, norm_const))

            updated_ids = cur.fetchall()
            if len(updated_ids) == 1:
                matched += 1
            elif len(updated_ids) == 0:
                unmatched.append((constituency, state, "no dim_mp match found"))
            else:
                # More than one dim_mp row matched this constituency+state -
                # flag it rather than silently updating multiple MPs with
                # the same party/status/terms, which could be wrong.
                ambiguous.append((constituency, state, len(updated_ids)))

        conn.commit()
        print(f"\nDone. {matched} dim_mp rows updated with party/status/terms.")

        if ambiguous:
            print(f"\n{len(ambiguous)} constituency+state pairs matched MULTIPLE dim_mp "
                  f"rows (updated all of them - please review these manually):")
            for constituency, state, count in ambiguous:
                print(f"  - {constituency!r} / {state!r} -> matched {count} dim_mp rows")

        if unmatched:
            print(f"\n{len(unmatched)} rows could NOT be matched — "
                  f"check these for constituency-name mismatches:")
            for constituency, state, reason in unmatched[:20]:
                print(f"  - {constituency!r} / {state!r} -> {reason}")
            if len(unmatched) > 20:
                print(f"  ...and {len(unmatched) - 20} more")

    except Exception as e:
        conn.rollback()
        print("\nSomething went wrong - no changes were saved. Error details:")
        print(e)

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
