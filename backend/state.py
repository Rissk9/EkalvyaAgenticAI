from typing import TypedDict


class AgentState(TypedDict):
    input: str
    data: dict
    decision: dict
    github_username: str
    leetcode_username: str
    summary: str
    output: str
