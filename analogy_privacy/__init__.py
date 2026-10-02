"""analogy-privacy: policy-invariant fictional analogues with a locally checkable verifier.

Work in progress. See README.md for status and for what the verifier does and
does not establish.
"""

from .invariance import (
    InvarianceReport,
    SensitivityReport,
    check_invariance,
    sensitivity_probe,
)
from .leak_audit import (
    LeakAuditReport,
    Severity,
    audit_leaks,
    check_direct_leaks,
    estimate_uniqueness,
)
from .roundtrip import (
    AuditRecord,
    ClearanceRefused,
    back_map,
    mark_cleared,
    new_record,
    release_to_cloud,
    sign_off,
)
from .schema import Case, CaseField, Role, project_decision_relevant
from .transformers import (
    OllamaTransformer,
    RuleBasedTransformer,
    TransformError,
    TransformResult,
    Transformer,
)

__version__ = "0.0.1"

__all__ = [
    "Case",
    "CaseField",
    "Role",
    "project_decision_relevant",
    "check_invariance",
    "sensitivity_probe",
    "InvarianceReport",
    "SensitivityReport",
    "audit_leaks",
    "check_direct_leaks",
    "estimate_uniqueness",
    "LeakAuditReport",
    "Severity",
    "AuditRecord",
    "ClearanceRefused",
    "new_record",
    "sign_off",
    "mark_cleared",
    "release_to_cloud",
    "back_map",
    "Transformer",
    "RuleBasedTransformer",
    "OllamaTransformer",
    "TransformResult",
    "TransformError",
]
