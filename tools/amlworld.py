import argparse
import csv
import datetime
import json
import os
import pathlib
import numpy as np
import duckdb
from bank_data import private_json, default_query, prepare_directory, finish_dataset
from column_data import write_bank

COLUMNS = dict(
    zip(
        ["ts", "fb", "fa", "tb", "ta", "received", "rc", "paid", "pc", "format", "tag"],
        ["VARCHAR"] * 11,
    )
)
ACCOUNT_COLUMNS = dict(
    zip(["bank_name", "bank", "account", "entity", "entity_name"], ["VARCHAR"] * 5)
)


def quoted(text):
    return "'" + str(text).replace("'", "''") + "'"


def query_csv(db, sql, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db.execute(f"COPY ({sql}) TO {quoted(path)} (HEADER,DELIMITER ',')")


def scalar(db, sql):
    return db.execute(sql).fetchone()[0]


def json_write(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def readcsv(path, columns):
    types = "{" + ",".join(quoted(k) + ":" + quoted(v) for k, v in columns.items()) + "}"
    return f"read_csv({quoted(path)},columns={types},header=true,auto_detect=false,delim=',',parallel=false,strict_mode=true,ignore_errors=false)"


def bank_expression(name, n):
    return f"(CAST('0x'||substr(sha256(CAST({name} AS VARCHAR)),1,16) AS UBIGINT)%{n})"


def database(raw_root, work, variant, threads):
    path = work / f"{variant}.duckdb"
    db = duckdb.connect(str(path))
    db.execute(f"SET threads={int(threads)}")
    db.execute("SET memory_limit='32GB'")
    db.execute("SET preserve_insertion_order=true")
    db.execute(f'SET temp_directory={quoted(work/(variant+"_tmp"))}')
    if scalar(db, "SELECT count(*) FROM information_schema.tables WHERE table_name='ready'"):
        return db
    if scalar(db, "SELECT count(*) FROM information_schema.tables WHERE table_name='raw'") == 0:
        db.execute(
            "CREATE TABLE raw AS SELECT row_number() OVER () AS ordinal,* FROM "
            f'{readcsv(raw_root/(variant+"_Trans.csv"),COLUMNS)}'
        )
    if (
        scalar(db, "SELECT count(*) FROM information_schema.tables WHERE table_name='account_map'")
        == 0
    ):
        db.execute(
            "CREATE TABLE accounts AS SELECT TRY_CAST(bank AS UBIGINT) bank,"
            "trim(account) account,trim(entity) entity FROM "
            f'{readcsv(raw_root/(variant+"_accounts.csv"),ACCOUNT_COLUMNS)}'
        )
        assert (
            scalar(
                db, "SELECT count(*) FROM accounts WHERE bank IS NULL OR account='' OR entity='' "
            )
            == 0
        )
        assert (
            scalar(
                db,
                "SELECT count(*) FROM (SELECT bank,account FROM accounts "
                "GROUP BY bank,account HAVING count(DISTINCT entity)>1)",
            )
            == 0
        ), "conflicting entity mapping"
        db.execute(
            "CREATE TABLE identities AS SELECT entity,"
            "row_number() OVER (ORDER BY entity) sid "
            "FROM (SELECT DISTINCT entity FROM accounts)"
        )
        db.execute(
            "CREATE TABLE account_map AS SELECT DISTINCT bank,account,sid "
            "FROM accounts JOIN identities USING(entity)"
        )
    t0 = scalar(
        db,
        "SELECT date_trunc('day',min(try_strptime(ts,['%Y/%m/%d %H:%M','%Y/%m/%d %H:%M:%S']))) FROM raw",
    )
    db.execute(
        "CREATE OR REPLACE TABLE parsed AS SELECT ordinal,"
        "try_strptime(ts,['%Y/%m/%d %H:%M','%Y/%m/%d %H:%M:%S']) time,"
        "TRY_CAST(fb AS UBIGINT) fb,trim(fa) fa,"
        "TRY_CAST(tb AS UBIGINT) tb,trim(ta) ta,"
        "TRY_CAST(TRY_CAST(paid AS DECIMAL(30,2))*100 AS BIGINT) paid,"
        "TRY_CAST(TRY_CAST(received AS DECIMAL(30,2))*100 AS BIGINT) received,"
        "rc,pc,TRY_CAST(tag AS INTEGER) tag,"
        "regexp_full_match(raw.paid,'[0-9]+([.][0-9]{1,2})?') paid_exact,"
        "regexp_full_match(raw.received,'[0-9]+([.][0-9]{1,2})?') received_exact FROM raw"
    )
    condition = (
        "rc='US Dollar' AND pc='US Dollar' AND time IS NOT NULL "
        "AND fb IS NOT NULL AND tb IS NOT NULL AND paid_exact AND received_exact "
        "AND paid>0 AND paid<1099511627776 AND received>0 "
        "AND received<1099511627776 AND tag IN (0,1)"
    )
    missing = scalar(
        db,
        "SELECT count(*) FROM parsed r "
        "LEFT JOIN account_map x ON r.fb=x.bank AND r.fa=x.account "
        "LEFT JOIN account_map y ON r.tb=y.bank AND r.ta=y.account "
        f"WHERE {condition} AND (x.sid IS NULL OR y.sid IS NULL)",
    )
    assert missing == 0, ("missing customer mapping", missing)
    db.execute(
        "CREATE OR REPLACE TABLE clean AS "
        "SELECT ordinal,time,fb,tb,paid,received,tag,x.sid payer,y.sid payee "
        "FROM parsed r JOIN account_map x ON r.fb=x.bank AND r.fa=x.account "
        f"JOIN account_map y ON r.tb=y.bank AND r.ta=y.account WHERE {condition}"
    )
    raw_count = scalar(db, "SELECT count(*) FROM raw")
    clean_count = scalar(db, "SELECT count(*) FROM clean")
    stats = {
        "variant": variant,
        "kind": "AMLworld",
        "description": "AMLworld realistic synthetic transactions",
        "raw_transactions": raw_count,
        "clean_transactions": clean_count,
        "protocol_input_rows": 2 * clean_count,
        "t0": t0.isoformat(),
        "time_unit": "relative_seconds",
        "bt": 29,
        "amount_unit": "exact cents",
        "amount_limit_exclusive": 1 << 40,
        "identity": "supplied entity ID; account key includes original bank ID",
        "accounts": scalar(db, "SELECT count(*) FROM account_map"),
        "entities_in_account_map": scalar(db, "SELECT count(*) FROM identities"),
        "active_customers": scalar(
            db,
            "SELECT count(*) FROM (SELECT payer sid FROM clean UNION SELECT payee sid FROM clean)",
        ),
        "currency_rule": (
            "both paid and received currencies exactly US Dollar; " "no FX conversion or clipping"
        ),
        "invalid_rule": "drop with explicit counts, never sample",
        "rejected_transactions_total": raw_count - clean_count,
        "currency_rejected": scalar(
            db, "SELECT count(*) FROM parsed WHERE NOT(rc='US Dollar' AND pc='US Dollar')"
        ),
        "invalid_time": scalar(db, "SELECT count(*) FROM parsed WHERE time IS NULL"),
        "invalid_bank": scalar(db, "SELECT count(*) FROM parsed WHERE fb IS NULL OR tb IS NULL"),
        "invalid_amount": scalar(
            db,
            "SELECT count(*) FROM parsed WHERE paid IS NULL OR received IS NULL "
            "OR NOT paid_exact OR NOT received_exact OR paid<=0 OR received<=0 "
            "OR paid>=1099511627776 OR received>=1099511627776",
        ),
        "invalid_tag": scalar(
            db, "SELECT count(*) FROM parsed WHERE tag IS NULL OR tag NOT IN (0,1)"
        ),
        "rejection_categories_can_overlap": True,
        "clean_min_time": scalar(db, "SELECT min(time) FROM clean"),
        "clean_max_time": scalar(db, "SELECT max(time) FROM clean"),
        "duckdb_version": duckdb.__version__,
    }
    stats["source_directory"] = str(raw_root.resolve())
    json_write(work / f"{variant}_manifest.json", stats)
    query_csv(
        db,
        "SELECT pc,rc,count(*) transactions FROM parsed GROUP BY pc,rc ORDER BY transactions DESC",
        work / f"{variant}_currencies.csv",
    )
    db.execute("CREATE TABLE ready AS SELECT true completed")
    return db


def counts_sql(begin, end):
    where = f"time>={quoted(begin)}::TIMESTAMP AND time<{quoted(end)}::TIMESTAMP"
    return (
        f"WITH activity AS (SELECT payee sid,1 inc,0 outc,tag FROM clean WHERE {where} "
        f"UNION ALL SELECT payer sid,0 inc,1 outc,tag FROM clean WHERE {where}) "
        "SELECT sid,sum(inc)::BIGINT nin,sum(outc)::BIGINT nout,max(tag) laundering "
        "FROM activity GROUP BY sid"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--variant", choices=["HI-Small", "HI-Medium", "HI-Large"], default="HI-Medium"
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--work", type=pathlib.Path)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--begin")
    parser.add_argument("--days", type=int)
    parser.add_argument("--subject", type=int)
    args = parser.parse_args()
    if (
        not 2 <= args.n <= 128
        or args.K < 1
        or args.threads < 1
        or (args.days is not None and args.days < 1)
    ):
        parser.error("Invalid dimensions")
    os.umask(0o077)
    source = args.source_dir.resolve()
    for suffix in ["_Trans.csv", "_accounts.csv"]:
        if not (source / (args.variant + suffix)).is_file():
            parser.error(f"Missing {args.variant + suffix}")
    target = args.output.resolve()
    if target.exists():
        parser.error("Use a fresh output directory")
    work = args.work.resolve() if args.work else target.parent / ".aml-cache"
    work.mkdir(parents=True, exist_ok=True)
    sources = {
        suffix: {
            "bytes": (source / (args.variant + suffix)).stat().st_size,
            "mtime_ns": (source / (args.variant + suffix)).stat().st_mtime_ns,
            "path": str(source / (args.variant + suffix)),
        }
        for suffix in ["_Trans.csv", "_accounts.csv"]
    }
    signature = work / (args.variant + "-sources.json")
    if signature.exists() and json.loads(signature.read_text()) != sources:
        parser.error("Cached preprocessing belongs to different sources; choose a fresh --work")
    signature.write_text(json.dumps(sources, indent=2) + "\n")
    db = database(source, work, args.variant, args.threads)
    try:
        begin = (
            datetime.datetime.fromisoformat(args.begin)
            if args.begin
            else scalar(db, "SELECT date_trunc('day',min(time)) FROM clean")
        )
        end = (
            begin + datetime.timedelta(days=args.days)
            if args.days
            else scalar(db, "SELECT date_trunc('day',max(time))+INTERVAL 1 DAY FROM clean")
        )
        if begin is None or end is None or end <= begin or (end - begin).total_seconds() >= 1 << 29:
            raise ValueError("Invalid public time window")
        db.execute("CREATE OR REPLACE TEMP TABLE window_counts AS " + counts_sql(begin, end))
        if args.subject is None:
            row = db.execute(
                "SELECT sid FROM window_counts WHERE laundering=1 AND nin<=? AND nout<=? ORDER BY sid LIMIT 1",
                [args.K, args.K],
            ).fetchone()
            if row is None:
                row = db.execute(
                    "SELECT sid FROM window_counts ORDER BY nin+nout DESC,sid LIMIT 1"
                ).fetchone()
        else:
            row = db.execute("SELECT sid FROM window_counts WHERE sid=?", [args.subject]).fetchone()
        if row is None:
            raise ValueError("No selected entity has transactions in the window")
        subject = int(row[0])
        path = prepare_directory(target)
        banks = []
        for bank in range(args.n):
            sql = (
                f"SELECT sid,dir,epoch(time-{quoted(begin)}::TIMESTAMP)::BIGINT t,a,cp "
                "FROM (SELECT ordinal,payer sid,1::UTINYINT dir,time,paid a,payee cp "
                f"FROM clean WHERE {bank_expression('fb',args.n)}={bank} "
                "UNION ALL SELECT ordinal,payee sid,0::UTINYINT dir,time,received a,payer cp "
                f"FROM clean WHERE {bank_expression('tb',args.n)}={bank}) "
                f"WHERE time>={quoted(begin)}::TIMESTAMP AND time<{quoted(end)}::TIMESTAMP "
                "ORDER BY ordinal,dir"
            )
            cols = db.execute(sql).fetchnumpy()
            count = write_bank(path / f"bank_{bank}.bjcol", bank, args.n, cols)
            banks.append(dict(bank=bank, file=f"bank_{bank}.bjcol", rows=count))
            del cols
        source_info = dict(
            kind="AMLworld",
            variant=args.variant,
            identity="entity",
            window_begin=begin.isoformat(),
            window_end_exclusive=end.isoformat(),
            currency="US Dollar",
            amount_unit="exact cents",
            time_unit="relative_seconds",
            bank_assignment="uint64_be(first 8 bytes SHA256(canonical decimal bank ID)) % n",
            source_directory=str(source),
            sampling=False,
        )
        finish_dataset(path, banks, source_info, subject, {"subject": subject}, K=args.K)
        query_csv(
            db, "SELECT * FROM window_counts ORDER BY sid", path / "private/entity_counts.csv"
        )
        print(
            json.dumps(dict(dataset=str(path), rows=sum(b["rows"] for b in banks), subject=subject))
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
