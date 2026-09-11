"""
Pulls structured entities (NCT ID, condition, intervention, endpoint text)
out of raw trial JSON / classified text sections.
"""
import re

from src.storage.models import EligibilityCriterion, Endpoint, Intervention, Trial

# Matches a top-level bullet at the start of a line: "* ", "1. ", "(1) "
_BULLET_PATTERN = re.compile(r"(?:^|\n)\s*(?:[*\-]|\(?\d+[.)])\s+")


def trial_from_ctgov_record(record: dict) -> Trial:
    """
    Map a raw ClinicalTrials.gov API v2 JSON record to a Trial dataclass.
    """
    protocol = record.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    conditions_module = protocol.get("conditionsModule", {})

    phases = design.get("phases", [])
    conditions = conditions_module.get("conditions", [])

    return Trial(
        nct_id=identification.get("nctId", ""),
        title=identification.get("briefTitle", ""),
        phase=", ".join(phases) if phases else None,
        condition="; ".join(conditions) if conditions else None,
        status=status.get("overallStatus"),
        source="clinicaltrials.gov",
        raw_json=record,
    )


def interventions_from_ctgov_record(record: dict) -> list[Intervention]:
    """Pull from protocolSection.armsInterventionsModule.interventions."""
    protocol = record.get("protocolSection", {})
    nct_id = protocol.get("identificationModule", {}).get("nctId", "")
    interventions = protocol.get("armsInterventionsModule", {}).get("interventions", [])

    return [
        Intervention(
            nct_id=nct_id,
            name=item.get("name", ""),
            intervention_type=item.get("type"),
        )
        for item in interventions
    ]


def endpoints_from_ctgov_record(record: dict) -> list[Endpoint]:
    """
    Pull from protocolSection.outcomesModule (primary/secondary).
    Deliberately skips `otherOutcomes` for this POC's scope.
    """
    protocol = record.get("protocolSection", {})
    nct_id = protocol.get("identificationModule", {}).get("nctId", "")
    outcomes = protocol.get("outcomesModule", {})

    endpoints: list[Endpoint] = []
    for outcome in outcomes.get("primaryOutcomes", []):
        endpoints.append(
            Endpoint(nct_id=nct_id, endpoint_type="primary", description=outcome.get("measure", ""))
        )
    for outcome in outcomes.get("secondaryOutcomes", []):
        endpoints.append(
            Endpoint(nct_id=nct_id, endpoint_type="secondary", description=outcome.get("measure", ""))
        )
    return endpoints


def _split_bullets(block: str) -> list[str]:
    """
    Split a block of free text into individual criterion strings.

    Splits only at top-level bullet/number markers ('*', '1.', '(1)') so
    wrapped continuation lines stay attached to the criterion they belong
    to, rather than becoming their own (meaningless) entries.
    """
    positions = [m.start() for m in _BULLET_PATTERN.finditer(block)]
    if not positions:
        text = block.strip()
        return [text] if text else []

    positions.append(len(block))
    criteria = []
    for start, end in zip(positions, positions[1:]):
        chunk = _BULLET_PATTERN.sub("", block[start:end], count=1)
        chunk = " ".join(chunk.split())  # collapse internal newlines/whitespace
        if chunk:
            criteria.append(chunk)
    return criteria


def eligibility_from_ctgov_record(record: dict) -> list[EligibilityCriterion]:
    """
    Parse the free-text eligibility criteria block into separate
    inclusion / exclusion EligibilityCriterion rows.

    This text is the messiest field in the whole record: bullet styles
    are inconsistent across trials and even within one trial. Rule-based
    splitting handles the common cases; an LLM-based extractor is the
    natural upgrade path once this needs to be robust across thousands
    of trials rather than a POC-scale sample.
    """
    protocol = record.get("protocolSection", {})
    nct_id = protocol.get("identificationModule", {}).get("nctId", "")
    raw_text = protocol.get("eligibilityModule", {}).get("eligibilityCriteria", "")

    if not raw_text:
        return []

    split_point = re.search(r"exclusion criteria:?", raw_text, flags=re.IGNORECASE)
    if split_point:
        inclusion_block = raw_text[: split_point.start()]
        exclusion_block = raw_text[split_point.end():]
    else:
        inclusion_block, exclusion_block = raw_text, ""

    inclusion_block = re.sub(r"(?i)^\s*inclusion criteria:?", "", inclusion_block)

    criteria: list[EligibilityCriterion] = []
    for text in _split_bullets(inclusion_block):
        criteria.append(EligibilityCriterion(nct_id=nct_id, criterion_type="inclusion", description=text))
    for text in _split_bullets(exclusion_block):
        criteria.append(EligibilityCriterion(nct_id=nct_id, criterion_type="exclusion", description=text))

    return criteria