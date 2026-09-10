"""
Pulls structured entities (NCT ID, condition, intervention, endpoint text)
out of raw trial JSON / classified text sections.
"""
from src.storage.models import EligibilityCriterion, Endpoint, Intervention, Trial


def trial_from_ctgov_record(record: dict) -> Trial:
    """
    Map a raw ClinicalTrials.gov API v2 JSON record to a Trial dataclass.
    TODO: pull nctId, briefTitle, phase, condition, overallStatus from
    the nested protocolSection structure.
    """
    raise NotImplementedError


def interventions_from_ctgov_record(record: dict) -> list[Intervention]:
    """TODO: pull from protocolSection.armsInterventionsModule.interventions"""
    raise NotImplementedError


def endpoints_from_ctgov_record(record: dict) -> list[Endpoint]:
    """TODO: pull from protocolSection.outcomesModule (primary/secondary)"""
    raise NotImplementedError


def eligibility_from_ctgov_record(record: dict) -> list[EligibilityCriterion]:
    """
    TODO: parse the free-text eligibility criteria block
    (protocolSection.eligibilityModule.eligibilityCriteria) into
    separate inclusion / exclusion lines.
    """
    raise NotImplementedError
