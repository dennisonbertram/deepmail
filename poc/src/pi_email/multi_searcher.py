"""Multi-account Gmail searcher.

Wraps multiple GmailSearcher instances (one per account) behind the
``Searcher`` protocol so the expansion loop sees a unified message
stream without knowing about accounts.
"""

from __future__ import annotations

from google.oauth2.credentials import Credentials

from .gmail_searcher import GmailSearcher
from .searcher import SearchBatch


class MultiAccountSearcher:
    """Searches across multiple authenticated Gmail accounts.

    Each account gets its own ``GmailSearcher``.  ``search_and_fetch``
    queries every account and merges the results into a single
    ``SearchBatch``.
    """

    def __init__(
        self,
        accounts: dict[str, Credentials],
        *,
        max_results_per_query: int = 500,
        batch_size: int = 50,
    ):
        self._searchers: dict[str, GmailSearcher] = {}
        per_account_max = max(1, max_results_per_query // max(1, len(accounts)))
        for email, creds in accounts.items():
            self._searchers[email] = GmailSearcher(
                creds,
                max_results_per_query=per_account_max,
                batch_size=batch_size,
            )
        self._total_max = max_results_per_query

    @property
    def quota_used(self) -> int:
        return sum(s.quota_used for s in self._searchers.values())

    def search(self, query: str) -> list[str]:
        ids: list[str] = []
        for searcher in self._searchers.values():
            ids.extend(searcher.search(query))
        return ids

    def fetch(self, msg_id: str):
        """Try each account until the message is found."""
        last_exc = None
        for searcher in self._searchers.values():
            try:
                return searcher.fetch(msg_id)
            except Exception as exc:
                last_exc = exc
        raise last_exc or RuntimeError(f"Message {msg_id} not found in any account")

    def search_and_fetch(self, query: str) -> SearchBatch:
        all_hits = []
        total_quota = 0
        total_retries = 0
        any_truncated = False
        errors: list[str] = []

        for email, searcher in self._searchers.items():
            try:
                batch = searcher.search_and_fetch(query)
                all_hits.extend(batch.hits)
                total_quota += batch.quota_units_used
                total_retries += batch.retry_count
                if batch.truncated:
                    any_truncated = True
                if batch.error:
                    errors.append(f"[{email}] {batch.error}")
            except Exception as exc:
                errors.append(f"[{email}] {exc}")

        error_str = "; ".join(errors) if errors else None

        return SearchBatch(
            query=query,
            hits=all_hits,
            quota_units_used=total_quota,
            retry_count=total_retries,
            truncated=any_truncated,
            error=error_str,
        )
