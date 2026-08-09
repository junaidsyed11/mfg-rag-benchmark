"""
Synthetic manufacturing document generator.

Uses Llama 3.3 70B via OpenRouter to generate realistic aerospace/manufacturing
documents (NCR, ECR, Supplier Audit, Corrective Action, Incident Reports).

Sampling parameters are loaded from config.yaml — see the synthetic.sampling
section for tuning notes.

Usage:
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --doc-type ncr --count 10  # quick test
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_api_key, load_config

# ---------------------------------------------------------------------------
# Domain vocabulary — grounding the LLM in realistic aerospace/manufacturing
# language so outputs don't drift into generic business writing
# ---------------------------------------------------------------------------

SUPPLIERS = [
    "Apex Precision Components", "Orbital Fasteners Ltd", "NovaTech Composites",
    "Sterling Hydraulics", "Meridian Aerospace Castings", "Titan Alloy Works",
    "PrecisionFlow Systems", "Atlas Machining Group", "Vanguard Coatings Inc",
    "CoreTech Forgings", "Stratos Avionics Supply", "Helix Tooling Solutions",
]

PART_FAMILIES = [
    ("PN-{n}-A", "Turbine blade, stage 2"),
    ("AS-{n}-B", "Hydraulic actuator assembly"),
    ("FR-{n}-C", "Fuselage frame bracket"),
    ("LG-{n}-D", "Landing gear strut fitting"),
    ("EN-{n}-E", "Engine mount bolt cluster"),
    ("AV-{n}-F", "Avionics bay panel"),
    ("FU-{n}-G", "Fuel line coupling"),
    ("CS-{n}-H", "Control surface hinge"),
]

DEFECT_TYPES = [
    "Dimensional out-of-tolerance",
    "Surface finish non-conformance",
    "Material certification missing",
    "Incorrect heat treat condition",
    "Porosity detected on X-ray",
    "Thread form deviation",
    "Coating adhesion failure",
    "Wrong alloy composition",
    "Burr on critical datum surface",
    "Torque specification exceeded",
]

DISPOSITIONS = [
    "Return to Supplier (RTS)",
    "Use As Is (UAI) — Engineering disposition required",
    "Rework in-house",
    "Scrap",
    "Conditional acceptance — reinspection at next assembly",
]

ROOT_CAUSES = [
    "Worn tooling — CNC lathe axis calibration drift",
    "Operator error — incorrect fixture setup",
    "Drawing revision not communicated to production floor",
    "Incoming material out of spec — supplier process control failure",
    "Environmental control failure — humidity exceedance in clean room",
    "Measurement system error — gauge R&R not current",
    "Process parameter drift — oven temperature control",
    "Incorrect traveller revision used",
]

CORRECTIVE_ACTIONS = [
    "Supplier to submit updated PPAP within 30 days",
    "100% incoming inspection implemented until supplier corrective action verified",
    "Engineering to issue drawing revision with tighter tolerance callout",
    "Gauge recalibration and MSA study required before next production run",
    "Supplier quality engineer site visit scheduled",
    "Process FMEA to be updated with new failure mode",
    "Training records to be reviewed and updated for affected operators",
]

ECR_REASONS = [
    "Weight reduction initiative — material substitution",
    "Obsolete component replacement",
    "Supplier discontinuation of current revision",
    "Fatigue life improvement",
    "Cost reduction — alternate manufacturing process",
    "Regulatory compliance update",
    "Field service feedback — recurring maintenance issue",
]

AUDIT_FINDINGS = [
    "Calibration records incomplete for 3 of 12 gauges audited",
    "Corrective action from previous audit not fully implemented",
    "First Article Inspection (FAI) documentation missing for 2 part numbers",
    "No documented process for handling customer returns",
    "Training matrix not current — 4 operators missing required certifications",
    "Control plan not updated to reflect latest drawing revision",
]


def random_part() -> tuple[str, str]:
    template, desc = random.choice(PART_FAMILIES)
    n = random.randint(10000, 99999)
    return template.format(n=n), desc


def random_date(start_year: int = 2022, end_year: int = 2025) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")


def random_id(prefix: str, year: int | None = None) -> str:
    yr = year or random.randint(2022, 2025)
    seq = random.randint(1, 9999)
    return f"{prefix}-{yr}-{seq:05d}"


# ---------------------------------------------------------------------------
# Prompt builders — each doc type gets its own prompt template.
# The structured fields are injected so the LLM writes a narrative that is
# CONSISTENT with the metadata, not contradictory to it.
# ---------------------------------------------------------------------------

def build_ncr_prompt(ctx: dict) -> str:
    return f"""You are a quality engineer at an aerospace manufacturing company.
Write a realistic Non-Conformance Report (NCR) narrative for the following event.
The narrative should be 200-350 words, written in professional technical English,
first person plural ("we identified", "inspection revealed").
Do NOT use bullet points. Write flowing paragraphs as a real engineer would.

Context:
- Part Number: {ctx['part_number']} ({ctx['part_desc']})
- Supplier: {ctx['supplier']}
- Defect Type: {ctx['defect_type']}
- Date Identified: {ctx['date']}
- Severity: {ctx['severity']}
- Root Cause: {ctx['root_cause']}
- Disposition: {ctx['disposition']}
- Corrective Action: {ctx['corrective_action']}

Write only the narrative text. No headings, no labels, no JSON."""


def build_ecr_prompt(ctx: dict) -> str:
    return f"""You are a design engineer at an aerospace company.
Write a realistic Engineering Change Request (ECR) narrative description for the
following proposed change. 180-280 words, professional technical English.
Flowing paragraphs, no bullet points.

Context:
- Part Number: {ctx['part_number']} ({ctx['part_desc']})
- Change Reason: {ctx['reason']}
- Proposed By: {ctx['engineer']}
- Date: {ctx['date']}
- Impact Areas: {ctx['impact']}

Write only the narrative. No headings, no labels."""


def build_audit_prompt(ctx: dict) -> str:
    return f"""You are a supplier quality engineer writing up a supplier audit report.
Write the findings narrative for the following audit. 200-320 words, professional,
factual tone. Flowing paragraphs.

Context:
- Supplier: {ctx['supplier']}
- Audit Date: {ctx['date']}
- Lead Auditor: {ctx['auditor']}
- Overall Rating: {ctx['rating']}
- Key Findings: {ctx['findings']}

Write only the narrative findings section."""


def build_corrective_action_prompt(ctx: dict) -> str:
    return f"""You are a quality engineer writing a Corrective Action Report (CAR).
Write the description and resolution narrative. 180-280 words, technical English,
past tense where actions are complete, future tense where scheduled.

Context:
- Related NCR: {ctx['ncr_id']}
- Part Number: {ctx['part_number']}
- Root Cause Confirmed: {ctx['root_cause']}
- Actions Taken: {ctx['actions']}
- Verification Method: {ctx['verification']}
- Target Close Date: {ctx['close_date']}

Write only the narrative."""


def build_incident_prompt(ctx: dict) -> str:
    return f"""You are a safety officer writing an aerospace manufacturing incident report.
Write the incident description and investigation summary. 200-300 words, factual,
passive or active voice mixed naturally. No bullet points.

Context:
- Incident Type: {ctx['incident_type']}
- Location: {ctx['location']}
- Date: {ctx['date']}
- Equipment Involved: {ctx['equipment']}
- Immediate Cause: {ctx['cause']}
- Corrective Action: {ctx['corrective_action']}

Write only the narrative."""


# ---------------------------------------------------------------------------
# Context generators — build the structured metadata dict for each doc type
# ---------------------------------------------------------------------------

ENGINEERS = ["J. Mitchell", "S. Patel", "A. Kowalski", "R. Thompson", "L. Chen", "M. Okafor"]
AUDITORS = ["K. Nakamura", "P. Rodriguez", "T. Williams", "F. Brennan"]
RATINGS = ["Satisfactory", "Conditional", "Unsatisfactory"]
SEVERITIES = ["Minor", "Major", "Critical"]
INCIDENT_TYPES = [
    "Foreign Object Debris (FOD) event",
    "Tooling failure during machining",
    "Handling damage to flight-critical part",
    "Calibration equipment out-of-date discovery",
    "Incorrect material kitted to work order",
]
LOCATIONS = ["Machine shop — Bay 4", "Assembly floor — Line 2", "Incoming inspection — Dock 7",
             "Paint booth — Station 3", "NDT lab"]
EQUIPMENT = ["CNC vertical machining centre", "Coordinate measuring machine (CMM)",
             "Hydraulic press brake", "Ultrasonic NDT system", "Torque wrench set"]
VERIFICATIONS = [
    "First Article Inspection on next production lot",
    "Process audit by SQE within 60 days",
    "Statistical process control chart review",
    "Customer source inspection at supplier",
]
IMPACTS = [
    "Assembly schedule, tooling, work instructions",
    "Drawing, BOM, procurement spec",
    "Test procedure, acceptance criteria",
    "Material specification, supplier approved list",
]


def make_ncr_context() -> dict:
    pn, desc = random_part()
    return {
        "doc_id": random_id("NCR"),
        "doc_type": "NCR",
        "part_number": pn,
        "part_desc": desc,
        "supplier": random.choice(SUPPLIERS),
        "defect_type": random.choice(DEFECT_TYPES),
        "severity": random.choice(SEVERITIES),
        "root_cause": random.choice(ROOT_CAUSES),
        "disposition": random.choice(DISPOSITIONS),
        "corrective_action": random.choice(CORRECTIVE_ACTIONS),
        "date": random_date(),
        "raised_by": random.choice(ENGINEERS),
    }


def make_ecr_context() -> dict:
    pn, desc = random_part()
    return {
        "doc_id": random_id("ECR"),
        "doc_type": "ECR",
        "part_number": pn,
        "part_desc": desc,
        "reason": random.choice(ECR_REASONS),
        "engineer": random.choice(ENGINEERS),
        "impact": random.choice(IMPACTS),
        "date": random_date(),
    }


def make_audit_context() -> dict:
    findings = random.sample(AUDIT_FINDINGS, k=random.randint(2, 4))
    return {
        "doc_id": random_id("AUD"),
        "doc_type": "supplier_audit",
        "supplier": random.choice(SUPPLIERS),
        "auditor": random.choice(AUDITORS),
        "rating": random.choice(RATINGS),
        "findings": "; ".join(findings),
        "date": random_date(),
    }


def make_corrective_action_context() -> dict:
    pn, _ = random_part()
    return {
        "doc_id": random_id("CAR"),
        "doc_type": "corrective_action",
        "ncr_id": random_id("NCR"),
        "part_number": pn,
        "root_cause": random.choice(ROOT_CAUSES),
        "actions": random.choice(CORRECTIVE_ACTIONS),
        "verification": random.choice(VERIFICATIONS),
        "close_date": random_date(2024, 2025),
        "date": random_date(),
    }


def make_incident_context() -> dict:
    return {
        "doc_id": random_id("INC"),
        "doc_type": "incident_report",
        "incident_type": random.choice(INCIDENT_TYPES),
        "location": random.choice(LOCATIONS),
        "equipment": random.choice(EQUIPMENT),
        "cause": random.choice(ROOT_CAUSES),
        "corrective_action": random.choice(CORRECTIVE_ACTIONS),
        "date": random_date(),
    }


CONTEXT_BUILDERS = {
    "ncr": (make_ncr_context, build_ncr_prompt),
    "ecr": (make_ecr_context, build_ecr_prompt),
    "supplier_audit": (make_audit_context, build_audit_prompt),
    "corrective_action": (make_corrective_action_context, build_corrective_action_prompt),
    "incident_report": (make_incident_context, build_incident_prompt),
}


# ---------------------------------------------------------------------------
# OpenRouter call
# ---------------------------------------------------------------------------

def call_llm(prompt: str, cfg: dict, api_key: str) -> str:
    s = cfg["synthetic"]["sampling"]
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/junaidsyed11/mfg-rag-benchmark",
        },
        json={
            "model": cfg["synthetic"]["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": s["temperature"],
            "top_p": s["top_p"],
            "frequency_penalty": s["frequency_penalty"],
            "max_tokens": s["max_tokens"],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(doc_type: str, count: int, cfg: dict, api_key: str, out_path: Path) -> None:
    context_fn, prompt_fn = CONTEXT_BUILDERS[doc_type]
    out_file = out_path / f"{doc_type}.jsonl"
    print(f"\n  Generating {count} {doc_type.upper()} documents → {out_file}")

    with open(out_file, "w") as f:
        for i in range(count):
            ctx = context_fn()
            prompt = prompt_fn(ctx)

            try:
                narrative = call_llm(prompt, cfg, api_key)
                doc = {**ctx, "text": narrative}
                f.write(json.dumps(doc) + "\n")
                print(f"    [{i+1:03d}/{count}] {ctx['doc_id']} ✓")
            except Exception as e:
                print(f"    [{i+1:03d}/{count}] ERROR: {e}")

            # Polite rate limiting — OpenRouter free tier is generous but not unlimited
            time.sleep(0.5)

    print(f"  Done — {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic manufacturing documents")
    parser.add_argument("--doc-type", choices=list(CONTEXT_BUILDERS.keys()),
                        help="Generate only this doc type (default: all)")
    parser.add_argument("--count", type=int, help="Override document count")
    args = parser.parse_args()

    cfg = load_config()
    api_key = get_api_key("OPENROUTER_API_KEY")
    out_path = Path(cfg["paths"]["synthetic_data"])
    out_path.mkdir(parents=True, exist_ok=True)

    counts = cfg["synthetic"]["n_documents"]

    if args.doc_type:
        doc_types = {args.doc_type: args.count or counts[args.doc_type]}
    else:
        doc_types = counts

    print(f"Model: {cfg['synthetic']['model']}")
    print(f"Temperature: {cfg['synthetic']['sampling']['temperature']}")
    print(f"Output: {out_path}/")

    for doc_type, count in doc_types.items():
        generate(doc_type, count, cfg, api_key, out_path)

    print("\nAll done.")


if __name__ == "__main__":
    main()
