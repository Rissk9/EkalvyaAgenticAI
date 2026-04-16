from typing import TypedDict, Optional


class AgentState(TypedDict):
    input: str
    data: dict
    decision: dict
    github_username: str
    leetcode_username: str

    # Target role & company for tailored advice
    role: Optional[str]
    company: Optional[str]

    summary: str
    output: str
