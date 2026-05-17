import json
import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)

"""
Dead Letter Queue (DLQ) Audit Logger
"""
class DeadLetterQueue:
    """Logs all orders rejected by the Risk Critic to an audit journal for post-mortem analysis."""
    def __init__(self, journal_path: str = "dlq_audit.json"):
        self.journal_path = journal_path
        
    def log_rejection(self, proposed_order: Dict, failure_reason: str):
        audit_payload = {
            "timestamp": int(time.time() * 1e9),
            "proposed_order": proposed_order,
            "rejection_reason": failure_reason
        }
        try:
            with open(self.journal_path, "a") as f:
                f.write(json.dumps(audit_payload) + "\n")
            logger.warning(f"DLQ AUDIT LOGGED -> {failure_reason}")
        except Exception as e:
            logger.error(f"Failed to write to DLQ journal: {e}")
