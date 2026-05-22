import json
import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)

class DeadLetterQueue:
    """Logs all orders rejected by the Risk Critic to an audit journal for post-mortem analysis."""
    def __init__(self, journal_path: str = "dlq_audit.json"):
        """
        Initializes DeadLetterQueue.

        Args:
            journal_path: Target file location to append json rejection lines.
        """
        self.journal_path = journal_path

    def log_rejection(self, proposed_order: Dict, failure_reason: str):
        """
        Appends rejected order details along with failure reasoning into audit ledger.

        Args:
            proposed_order: Failed order payload dictionary.
            failure_reason: Reason the order was blocked by Critic guardrails.
        """
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
