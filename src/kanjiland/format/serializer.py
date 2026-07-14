"""Document -> wire-format serializer (M0).

Contract:
    serialize(doc: Document) -> str
    - One record per line (newline after each RECORD_END).
    - Must satisfy: parse(serialize(doc)) == doc.
    - Raises ValueError if any content field contains a reserved codepoint
      (serializer refuses to produce invalid wire text).
"""

from .records import Document


def serialize(doc: Document) -> str:
    raise NotImplementedError("M0: implement with Claude Code (see tests/test_format.py)")
