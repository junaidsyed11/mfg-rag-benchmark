"""
Build a golden evaluation dataset from structured synthetic document metadata.

Instead of manual labelling, we derive ground-truth query/answer pairs
directly from the metadata fields that Llama 70B used when generating each
document. For example:
  - defect_type = "Porosity detected on X-ray"
    → query: "Which NCRs involved porosity defects?"
    → relevant_ids: all doc_ids where defect_type matches

This gives us deterministic, verifiable ground truth for free.

Output: data/golden.jsonl — one query per line:
  {
    "query_id": "ncr_defect_porosity",
    "query": "Which NCRs involved porosity defects?",
    "doc_type": "NCR",
    "relevant_doc_ids": ["NCR-2024-05493", ...]
  }

Usage:
    python scripts/build_golden.py
    python scripts/build_golden.py --stats
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import load_config

SYNTHETIC_QUERIES = [
    # NCR — defect type queries
    {
        "query_id": "ncr_defect_porosity",
        "query": "Which NCRs involved porosity defects?",
        "doc_type": "NCR",
        "field": "defect_type",
        "match": "porosity",
    },
    {
        "query_id": "ncr_defect_coating",
        "query": "Which parts had coating adhesion failures?",
        "doc_type": "NCR",
        "field": "defect_type",
        "match": "coating adhesion",
    },
    {
        "query_id": "ncr_defect_dimensional",
        "query": "Find non-conformance reports for dimensional out-of-tolerance defects",
        "doc_type": "NCR",
        "field": "defect_type",
        "match": "dimensional",
    },
    {
        "query_id": "ncr_defect_heat_treat",
        "query": "Which NCRs were raised for incorrect heat treat conditions?",
        "doc_type": "NCR",
        "field": "defect_type",
        "match": "heat treat",
    },
    {
        "query_id": "ncr_defect_material_cert",
        "query": "Which non-conformances involved missing material certifications?",
        "doc_type": "NCR",
        "field": "defect_type",
        "match": "material certification",
    },
    # NCR — disposition queries
    {
        "query_id": "ncr_disposition_scrap",
        "query": "Which non-conforming parts were scrapped?",
        "doc_type": "NCR",
        "field": "disposition",
        "match": "scrap",
    },
    {
        "query_id": "ncr_disposition_rts",
        "query": "Which parts were returned to supplier?",
        "doc_type": "NCR",
        "field": "disposition",
        "match": "return to supplier",
    },
    {
        "query_id": "ncr_disposition_uai",
        "query": "Which NCRs resulted in a use-as-is disposition?",
        "doc_type": "NCR",
        "field": "disposition",
        "match": "use as is",
    },
    # NCR — root cause queries
    {
        "query_id": "ncr_rootcause_operator",
        "query": "Which defects were caused by operator error?",
        "doc_type": "NCR",
        "field": "root_cause",
        "match": "operator error",
    },
    {
        "query_id": "ncr_rootcause_tooling",
        "query": "Which NCRs had worn tooling as the root cause?",
        "doc_type": "NCR",
        "field": "root_cause",
        "match": "worn tooling",
    },
    {
        "query_id": "ncr_rootcause_drawing",
        "query": "Which non-conformances were caused by drawing revision not communicated?",
        "doc_type": "NCR",
        "field": "root_cause",
        "match": "drawing revision",
    },
    # NCR — severity
    {
        "query_id": "ncr_severity_critical",
        "query": "Show me all critical severity non-conformance reports",
        "doc_type": "NCR",
        "field": "severity",
        "match": "critical",
    },
    # NCR — supplier
    {
        "query_id": "ncr_supplier_apex",
        "query": "What NCRs were raised against Apex Precision Components?",
        "doc_type": "NCR",
        "field": "supplier",
        "match": "apex precision",
    },
    {
        "query_id": "ncr_supplier_atlas",
        "query": "Find all non-conformances from Atlas Machining Group",
        "doc_type": "NCR",
        "field": "supplier",
        "match": "atlas machining",
    },
    {
        "query_id": "ncr_supplier_orbital",
        "query": "Which parts from Orbital Fasteners had defects?",
        "doc_type": "NCR",
        "field": "supplier",
        "match": "orbital fasteners",
    },
    # NCR — part family
    {
        "query_id": "ncr_part_turbine",
        "query": "Which NCRs involved turbine blade defects?",
        "doc_type": "NCR",
        "field": "part_desc",
        "match": "turbine blade",
    },
    {
        "query_id": "ncr_part_landing_gear",
        "query": "Which non-conformances affected landing gear components?",
        "doc_type": "NCR",
        "field": "part_desc",
        "match": "landing gear",
    },
    # Corrective action queries (field is "actions", no supplier field)
    {
        "query_id": "car_action_drawing",
        "query": "Which corrective actions required an engineering drawing revision?",
        "doc_type": "corrective_action",
        "field": "actions",
        "match": "drawing revision",
    },
    {
        "query_id": "car_rootcause_material",
        "query": "Which corrective actions were triggered by incoming material out of spec?",
        "doc_type": "corrective_action",
        "field": "root_cause",
        "match": "incoming material",
    },
    {
        "query_id": "car_rootcause_operator",
        "query": "Which corrective actions had operator error as the root cause?",
        "doc_type": "corrective_action",
        "field": "root_cause",
        "match": "operator error",
    },
    # Supplier audit queries
    {
        "query_id": "audit_supplier_novatech",
        "query": "Find supplier audit reports for NovaTech Composites",
        "doc_type": "supplier_audit",
        "field": "supplier",
        "match": "novatech",
    },
    {
        "query_id": "audit_supplier_sterling",
        "query": "What were the findings from Sterling Hydraulics audits?",
        "doc_type": "supplier_audit",
        "field": "supplier",
        "match": "sterling hydraulics",
    },
    # ECR queries
    {
        "query_id": "ecr_reason_weight",
        "query": "Which engineering changes were driven by weight reduction?",
        "doc_type": "ECR",
        "field": "reason",
        "match": "weight reduction",
    },
    {
        "query_id": "ecr_reason_obsolete",
        "query": "Which ECRs were raised to replace obsolete components?",
        "doc_type": "ECR",
        "field": "reason",
        "match": "obsolete",
    },
    # Incident report queries
    {
        "query_id": "ir_type_fod",
        "query": "Find incident reports involving Foreign Object Debris events",
        "doc_type": "incident_report",
        "field": "incident_type",
        "match": "foreign object debris",
    },
    {
        "query_id": "ir_type_tooling_failure",
        "query": "Which incidents involved tooling failure during machining?",
        "doc_type": "incident_report",
        "field": "incident_type",
        "match": "tooling failure",
    },
    {
        "query_id": "ir_cause_operator",
        "query": "Which incident reports were caused by operator error?",
        "doc_type": "incident_report",
        "field": "cause",
        "match": "operator error",
    },
]

# Maps doc_type values in the JSONL to file names
_DOCTYPE_TO_FILE = {
    "NCR":                "ncr.jsonl",
    "ECR":                "ecr.jsonl",
    "corrective_action":  "corrective_action.jsonl",
    "supplier_audit":     "supplier_audit.jsonl",
    "incident_report":    "incident_report.jsonl",
}


def load_docs(synthetic_dir: Path) -> dict[str, list[dict]]:
    """Load all synthetic docs keyed by doc_type."""
    docs: dict[str, list[dict]] = {}
    for doc_type, filename in _DOCTYPE_TO_FILE.items():
        path = synthetic_dir / filename
        if not path.exists():
            continue
        with open(path) as f:
            docs[doc_type] = [json.loads(line) for line in f if line.strip()]
    return docs


def build_golden(docs: dict[str, list[dict]]) -> list[dict]:
    """Generate query/relevant_doc_ids pairs from structured metadata."""
    golden = []
    for spec in SYNTHETIC_QUERIES:
        doc_list = docs.get(spec["doc_type"], [])
        match = spec["match"].lower()

        relevant_ids = [
            d["doc_id"]
            for d in doc_list
            if match in str(d.get(spec["field"], "")).lower()
        ]

        if len(relevant_ids) < 1:
            continue  # Skip queries with no relevant docs

        golden.append({
            "query_id":        spec["query_id"],
            "query":           spec["query"],
            "doc_type":        spec["doc_type"],
            "relevant_doc_ids": relevant_ids,
            "n_relevant":      len(relevant_ids),
        })

    return golden


def main() -> None:
    parser = argparse.ArgumentParser(description="Build golden evaluation dataset")
    parser.add_argument("--stats", action="store_true", help="Print stats and exit")
    args = parser.parse_args()

    cfg = load_config()
    synthetic_dir = Path(cfg["paths"]["synthetic_data"])
    out_path = Path("data/golden.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    docs = load_docs(synthetic_dir)
    total_docs = sum(len(v) for v in docs.values())
    print(f"Loaded {total_docs} documents across {len(docs)} types")

    golden = build_golden(docs)

    if args.stats:
        print(f"\n{'Query ID':<35} {'Type':<20} {'Relevant':>8}")
        print("-" * 65)
        for q in golden:
            print(f"  {q['query_id']:<33} {q['doc_type']:<20} {q['n_relevant']:>8}")
        print("-" * 65)
        print(f"  Total queries: {len(golden)}")
        return

    with open(out_path, "w") as f:
        for q in golden:
            f.write(json.dumps(q) + "\n")

    print(f"Written {len(golden)} queries to {out_path}")
    for q in golden:
        print(f"  {q['query_id']:<35} {q['n_relevant']:>3} relevant docs")


if __name__ == "__main__":
    main()
