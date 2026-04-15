"""
LangChain tools — resume, GitHub, LeetCode.
"""
import datetime
import requests
from concurrent.futures import ThreadPoolExecutor

from langchain.tools import tool

from backend.dependencies import get_retriever


@tool
def resume_tool(query: str) -> str:
    """Fetch relevant resume context for a given query."""
    print("➡️ Resume tool running")
    retriever = get_retriever()
    docs = retriever.invoke(query)[:2]
    context = "\n\n".join([d.page_content[:300] for d in docs])
    return context


@tool
def github_tool(username: str) -> str:
    """Analyze GitHub profile for activity, consistency, and project depth."""
    print("🐙 GitHub tool running!")
    try:
        rate = requests.get("https://api.github.com/rate_limit", timeout=5).json()
        remaining = rate["rate"]["remaining"]
        print(f"GitHub API calls remaining: {remaining}")
        if remaining < 10:
            return f"GitHub API rate limit nearly exhausted ({remaining} remaining). Try again later or add a GitHub token."

        repos_url = f"https://api.github.com/users/{username}/repos"
        repos = requests.get(repos_url, timeout=10).json()
        repo_count = len(repos)
        languages: dict = {}
        repos_to_check = repos[:5]

        def fetch_commits(repo):
            repo_name = repo["name"]
            commits_url = f"https://api.github.com/repos/{username}/{repo_name}/commits"
            commits = requests.get(commits_url, timeout=10).json()
            return repo, commits

        total_commits = 0
        recent_active_repos = 0

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(fetch_commits, repos_to_check))

        for repo, commits in results:
            lang = repo.get("language")
            if lang:
                languages[lang] = languages.get(lang, 0) + 1
            if isinstance(commits, list):
                total_commits += len(commits)
                if commits:
                    last_date = commits[0]["commit"]["author"]["date"]
                    last_date = datetime.datetime.fromisoformat(last_date.replace("Z", "+00:00"))
                    if (datetime.datetime.now(datetime.timezone.utc) - last_date).days < 30:
                        recent_active_repos += 1

        consistency = "high" if total_commits > 50 else "medium" if total_commits > 20 else "low"
        depth = "high" if repo_count > 20 else "medium" if repo_count > 8 else "low"
        top_languages = sorted(languages.items(), key=lambda x: x[1], reverse=True)[:3]

        return f"""
        GitHub Analysis:
        - Username: {username}
        - Total Repositories: {repo_count}
        - Sampled Commits: {total_commits}
        - Recently Active Repos: {recent_active_repos}
        - Consistency Level: {consistency}
        - Project Depth: {depth}
        - Top Languages: {top_languages}
        """
    except requests.exceptions.Timeout:
        return "GitHub API timed out. Try again."
    except Exception as e:
        return f"GitHub data unavailable: {str(e)}"


@tool
def leetcode_tool(username: str) -> str:
    """Fetch LeetCode profile stats for career evaluation."""
    print("🟡 LeetCode tool running!")
    url = "https://leetcode.com/graphql/"
    headers = {"Content-Type": "application/json", "Referer": "https://leetcode.com"}

    queries = {
        "solved": {
            "query": """
            query userProblemsSolved($username: String!) {
                matchedUser(username: $username) {
                    submitStatsGlobal {
                        acSubmissionNum { difficulty count }
                    }
                }
            }""",
            "variables": {"username": username},
        },
        "skills": {
            "query": """
            query skillStats($username: String!) {
                matchedUser(username: $username) {
                    tagProblemCounts {
                        advanced { tagName problemsSolved }
                        intermediate { tagName problemsSolved }
                        fundamental { tagName problemsSolved }
                    }
                }
            }""",
            "variables": {"username": username},
        },
        "contest": {
            "query": """
            query userContestRankingInfo($username: String!) {
                userContestRanking(username: $username) {
                    attendedContestsCount rating globalRanking topPercentage
                }
            }""",
            "variables": {"username": username},
        },
        "calendar": {
            "query": """
            query userProfileCalendar($username: String!) {
                matchedUser(username: $username) {
                    userCalendar { streak totalActiveDays }
                }
            }""",
            "variables": {"username": username},
        },
    }

    try:
        results = {}
        for key, payload in queries.items():
            res = requests.post(url, json=payload, headers=headers, timeout=10)
            results[key] = res.json().get("data", {})

        ac = results["solved"].get("matchedUser", {}).get("submitStatsGlobal", {}).get("acSubmissionNum", [])
        easy   = next((x["count"] for x in ac if x["difficulty"] == "Easy"), 0)
        medium = next((x["count"] for x in ac if x["difficulty"] == "Medium"), 0)
        hard   = next((x["count"] for x in ac if x["difficulty"] == "Hard"), 0)
        total  = easy + medium + hard

        if total > 300:   dsa_level = "expert"
        elif total > 150: dsa_level = "strong"
        elif total > 75:  dsa_level = "moderate"
        elif total > 25:  dsa_level = "beginner"
        else:             dsa_level = "weak"

        contest = results["contest"].get("userContestRanking") or {}
        cal     = results["calendar"].get("matchedUser", {}).get("userCalendar", {})
        tag_data = results["skills"].get("matchedUser", {}).get("tagProblemCounts", {})

        advanced = sorted(tag_data.get("advanced", []),     key=lambda x: x["problemsSolved"], reverse=True)[:3]
        intermed = sorted(tag_data.get("intermediate", []), key=lambda x: x["problemsSolved"], reverse=True)[:3]
        fund     = sorted(tag_data.get("fundamental", []),  key=lambda x: x["problemsSolved"], reverse=True)[:3]

        return f"""
        LeetCode Analysis:
        - Username: {username}
        - Total Solved: {total} (Easy: {easy} | Medium: {medium} | Hard: {hard})
        - DSA Level: {dsa_level}
        - Contest Rating: {contest.get('rating', 'N/A')}
        - Global Ranking: {contest.get('globalRanking', 'N/A')}
        - Top Percentage: {contest.get('topPercentage', 'N/A')}%
        - Contests Attended: {contest.get('attendedContestsCount', 0)}
        - Current Streak: {cal.get('streak', 0)} days
        - Total Active Days: {cal.get('totalActiveDays', 0)}
        - Top Advanced Topics: {[x['tagName'] for x in advanced]}
        - Top Intermediate Topics: {[x['tagName'] for x in intermed]}
        - Top Fundamental Topics: {[x['tagName'] for x in fund]}
        """
    except requests.exceptions.Timeout:
        return "LeetCode API timed out."
    except Exception as e:
        return f"LeetCode data unavailable: {str(e)}"
