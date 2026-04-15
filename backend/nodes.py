"""
LangGraph nodes: decision, tool_node, response.
"""
import json

from backend.state import AgentState
from backend.dependencies import get_llm
from backend.tools import resume_tool, github_tool, leetcode_tool

# ---------------------------------------------------------------------------
# Mentor persona (kept here so the whole LLM behaviour is in one file)
# ---------------------------------------------------------------------------
MENTOR_PERSONA = """
You are "Pathfinder" — an experienced Indian career mentor who has guided 500+ students
from Tier-2 and Tier-3 colleges across India into meaningful tech careers. The current month is April
and the year is 2026. You will always provide roadmaps with respect to this month and year.

YOUR BACKGROUND:
- You understand the reality of students from cities like Bhopal, Coimbatore, Nagpur,
  Jaipur, Vizag, Trichy, Indore — not just Bangalore and Hyderabad and delhi.
- You know that most of these students don't have IIT/NIT privilege, alumni networks,
  or referral pipelines. They have to fight harder and smarter.
- You've seen students with raw talent but zero guidance — no one told them what to
  learn, what to build, or how to present themselves.
- But you should not continously poke them wtih the fact that they belong to tier 2 or 3 city
- Dont mention the rating of their college and indulge in any direct comparision with the top 
  colleges in india
- Be constructive an dempathize like a mentor, by providing motivational stories that dont just 
  concern placements but show determination and persistence of life in general
- Dont give motivation in every answer, keep some anaswers short and give detailed analysis
  ONLY WHEN REQUESTED

YOUR PRINCIPLES:
1. BE BRUTALLY HONEST but never discouraging. Tell them where they stand, but always
   show them the path forward.
2. NEVER assume access to expensive resources, paid courses, or premium tools.
   Recommend FREE or affordable alternatives (YouTube, freeCodeCamp, GFG, Striver's
   sheet, LeetCode free tier, open-source contributions).
3. THINK in terms of the INDIAN JOB MARKET specifically:
   - Placement drives, off-campus hiring, referrals via LinkedIn
   - Service companies (TCS, Infosys, Wipro, Cognizant) as stepping stones, not dead ends
   - Product companies (Razorpay, Zerodha, PhonePe, CRED, Flipkart) as realistic goals
   - MAANG/FAANG as aspirational but achievable with the right prep
   - Startup ecosystem (YC India, Bharat startups) as alternative paths
   - Government exams (GATE for PSUs/M.Tech) as valid options, not fallbacks
4. UNDERSTAND their constraints:
   - Many work part-time or have family pressure for quick placements
   - English fluency varies — don't use jargon without explaining it
   - Internet speed and laptop specs might limit what tools they can use
   - Peer pressure from friends getting placed at mass recruiters
5. FOCUS on what actually gets you hired in India:
   - DSA (most Indian interviews are DSA-heavy, especially service + product companies)
   - Projects that solve REAL Indian problems (UPI-based, Aadhaar, agriculture, edtech)
   - Competitive programming profiles (CodeChef, Codeforces, LeetCode ratings)
   - GitHub consistency > GitHub perfection
   - LinkedIn presence (Indian recruiters live on LinkedIn)
   - Communication skills (many Tier-2/3 grads lose offers here)
6. USE INDIAN CONTEXT in examples:
   - Reference Indian company hiring patterns and CTC ranges
   - Mention Indian tech communities (GDG, GDSC, local meetups, Twitter/X tech circles)
   - Reference relevant Indian certifications and programs (NPTEL, AICTE internships,
     Smart India Hackathon, GSoC as a differentiator)
   - Be aware of the service-vs-product company mindset shift
7. EMOTIONAL QUERIES:
   - Become a counsellor and empathize with the user
   - Goal is to make the conversation addictive
   - Make the user attach with you and understand each emotion well
   - Empathize, Sympathize, Motivate and Encourage

YOUR TONE:
- Like a supportive elder sibling or senior who made it out and came back to help
- Direct, no sugarcoating, but always constructive
- Use relatable language — you can be slightly informal when it helps
- Celebrate small wins (first PR merged, first LeetCode medium solved, first internship)
"""


# ---------------------------------------------------------------------------
# Helper: update conversation summary
# ---------------------------------------------------------------------------
def update_summary(old_summary: str, query: str, response: str) -> str:
    llm = get_llm()
    prompt = f"""
    Update the conversation summary.
    Summarize the conversation concisely in 3-5 bullet points.

    Previous Summary:
    {old_summary}

    New Interaction:
    User: {query}
    AI: {response}

    Updated Summary:

    Return the answer in the following structured format:

    ###Question
    ( clear answers, maximum 4 lines)
    """
    return llm.invoke(prompt).content


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
def decision_node(state: AgentState) -> dict:
    query = state["input"].lower().strip()

    # --- 0. GREETINGS (check first so they never fall through to off-topic) ---
    greeting_triggers = [
        "hi", "hello", "hey", "sup", "yo", "hola", "namaste",
        "good morning", "good afternoon", "good evening", "good night",
        "what's up", "whats up", "how are you", "how r u", "how are u",
        "who are you", "who r u", "what can you do", "what do you do",
        "introduce yourself", "start", "begin"
    ]

    is_greeting = any(query == g or query.startswith(g + " ") or query.startswith(g + ",") or query.startswith(g + "!") for g in greeting_triggers)

    if is_greeting:
        print("👋 Greeting detected")
        return {
            "decision": {
                "use_resume": False,
                "use_github": False,
                "use_leetcode": False,
                "off_topic": False,
                "emotional": False,
                "greeting": True
            }
        }

    # --- 1. CHECK CAREER RELEVANCE FIRST ---
    career_keywords = [
        # Resume & profile
        "resume", "skill", "experience", "eligible", "background",
        "profile", "education", "project", "internship", "job",
        "role", "fit", "evaluate", "analyze", "candidate",
        # GitHub & code
        "github", "code", "repo", "build", "commit",
        "consistency", "activity", "coding", "contributions",
        # Career planning
        "improve", "roadmap", "plan", "career", "hire",
        "company", "interview", "dsa", "leetcode", "prepare",
        "placement", "campus", "offer", "ctc", "salary",
        "startup", "freelance", "portfolio", "certificate",
        "learn", "course", "mentor", "guide",
        "strength", "weakness", "gap", "ready", "crack",
        # Companies
        "google", "microsoft", "amazon", "flipkart", "razorpay",
        "tcs", "infosys", "wipro", "sde", "developer",
        # Tech domains
        "frontend", "backend", "fullstack", "devops", "ml", "ai",
        "web", "app", "database", "cloud", "deploy",
        "validate", "idea", "product", "market", "feature",
        "tech stack", "architecture", "mvp", "pitch",
        # Student & year-based queries (conversational)
        "first year", "second year", "third year", "final year",
        "1st year", "2nd year", "3rd year", "4th year",
        "fresher", "student", "college", "semester", "btech", "b.tech",
        "should i", "is it worth", "worth it", "thinking about",
        "is interning", "do i need", "when should", "how do i start",
        "what should i", "good idea", "bad idea", "advice",
        "programming", "language", "python", "java", "javascript",
        "open source", "gsoc", "hackathon", "competition",
        "off campus", "on campus", "package", "lpa", "work"
    ]

    is_career = any(w in query for w in career_keywords)

    # --- If career-related → route normally ---
    if is_career:
        resume_keywords = [
            "resume", "skill", "experience", "eligible", "background",
            "profile", "education", "project", "internship", "job",
            "role", "fit", "evaluate", "analyze", "candidate"
        ]

        github_keywords = [
            "github", "code", "repo", "build", "commit",
            "consistency", "activity", "coding", "contributions"
        ]

        leetcode_keywords = [
            "leetcode", "dsa", "competitive", "problem solving",
            "algorithms", "data structures", "contest", "rating"
        ]

        full_eval_keywords = [
            "evaluate", "analyze", "google", "microsoft", "amazon",
            "crack", "eligible", "fit for", "ready for", "sde", "internship"
        ]

        use_resume = any(w in query for w in resume_keywords)
        use_github = any(w in query for w in github_keywords)
        use_leetcode = any(w in query for w in leetcode_keywords)

        # Full eval → force ALL tools
        if any(w in query for w in full_eval_keywords):
            use_resume = True
            use_github = True
            use_leetcode = True

        if not use_resume and not use_github:
            use_resume = True

        print(f"🧠 Decision → resume: {use_resume}, github: {use_github}, leetcode: {use_leetcode}")
        return {
            "decision": {
                "use_resume": use_resume,
                "use_github": use_github,
                "use_leetcode": use_leetcode,
                "off_topic": False,
                "emotional": False
            }
        }

    # --- 2. CHECK EMOTIONAL (only if NOT career) ---
    emotional_keywords = [
        "feel lost", "i feel", "confused", "stuck", "hopeless",
        "scared", "anxious", "depressed", "frustrated", "overwhelmed",
        "directionless", "don't know what to do", "no idea what",
        "what do i do", "help me", "failing", "give up", "giving up",
        "worthless", "falling behind", "everyone else is",
        "comparison", "so much pressure", "stressed out",
        "dropout", "backlog", "arrear", "no placements",
        "keep getting rejected", "impostor", "not good enough",
        "wasting my time", "regret", "family pressure",
        "parents are disappointed", "demotivated", "lonely",
        "nobody guides me", "no one to guide", "i'm lost"
    ]

    is_emotional = any(w in query for w in emotional_keywords)

    if is_emotional:
        print("💛 Emotional query — mentor mode activated")
        return {
            "decision": {
                "use_resume": False,
                "use_github": False,
                "use_leetcode": False,
                "off_topic": False,
                "emotional": True
            }
        }

    # --- 3. OFF-TOPIC ---
    print("🚫 Off-topic query detected")
    return {
        "decision": {
            "use_resume": False,
            "use_github": False,
            "use_leetcode": False,
            "off_topic": True,
            "emotional": False
        }
    }


def tool_node(state: AgentState) -> dict:
    user_input = state["input"]
    decision = state["decision"]
    data = {}

    if decision.get("use_resume"):
        raw = resume_tool.invoke(user_input)
        try:
            data["resume"] = json.loads(raw)
        except Exception:
            data["resume"] = raw

    if decision.get("use_github"):
        username = state.get("github_username")
        data["github"] = github_tool.invoke(username) if username else "GitHub username not provided"

    if decision.get("use_leetcode"):
        username = state.get("leetcode_username")
        data["leetcode"] = leetcode_tool.invoke(username) if username else "LeetCode username not provided"

    return {"data": data}


def response_node(state: AgentState) -> dict:
    llm = get_llm()
    user_input = state["input"]
    data = state.get("data", {})
    summary = state.get("summary", "")
    decision = state.get("decision", {})

    # --- GREETING ---
    if decision.get("greeting"):
        greeting_response = llm.invoke(f"""
            {MENTOR_PERSONA}

            The user just greeted you with: "{user_input}"

            Respond with a warm, friendly, and brief greeting (2-4 lines max).
            - Introduce yourself as Pathfinder naturally dont mention tier or anything
            - Say something encouraging and genuine dont mention cities
            - End with ONE open question to understand what they need help with
              (e.g., their year of study, what topic they want to explore)
            Keep it conversational — like a friendly senior, not a formal assistant.
        """).content
        new_summary = update_summary(summary, user_input, greeting_response)
        return {"output": greeting_response, "summary": new_summary}

    # --- OFF-TOPIC ---
    if decision.get("off_topic"):
        rejection = (
            "🚫 **That's outside my scope!**\n\n"
            "I'm Pathfinder — your career mentor for tech placements, "
            "resume building, GitHub improvement, and interview prep.\n\n"
            "I can help you with:\n"
            "- 📄 Resume evaluation\n"
            "- 💻 GitHub profile analysis\n"
            "- 🎯 Company targeting (TCS to MAANG)\n"
            "- 📚 DSA/project roadmaps\n"
            "- 🗣️ Interview preparation\n\n"
            "👉 Try asking something like: *\"Am I eligible for Razorpay SDE role?\"* "
            "or *\"How do I improve my GitHub?\"*"
        )
        return {"output": rejection}

    # --- EMOTIONAL ---
    if decision.get("emotional"):
        empathy_response = llm.invoke(f"""
            {MENTOR_PERSONA}
            The student just said: "{user_input}"
            They are feeling vulnerable, lost, or overwhelmed. This is NOT a technical question —
            this is a human moment. Respond like the elder sibling they never had.
            RULES:
            1. ACKNOWLEDGE their feelings first. Don't jump to advice immediately.
            2. NORMALIZE it — remind them that lakhs of students from Tier-2/3 colleges
            feel this way. They are not alone. Many successful engineers started exactly
            where they are now.
            3. Share a BRIEF relatable perspective — many people from small cities with no
            guidance have made it. No IIT tag needed.
            4. GENTLY pivot to ONE small actionable step they can take TODAY.
            Not a full roadmap — just one tiny thing to build momentum.
            Examples: "Solve one easy LeetCode problem today", "Update your LinkedIn headline",
            "Push one commit to GitHub today"
            5. End with genuine encouragement. Not generic motivation — something specific
            to what they said.
            Keep it warm, real, and under 15 lines. No markdown headers.
            Talk like a person, not a system.
            Conversation so far: {summary}
            """).content
        followup = "Whenever you're ready, I'm here. Want to start with something small — like a quick resume check or a beginner-friendly project idea?"
        new_summary = update_summary(summary, user_input, empathy_response)
        return {"output": empathy_response + "\n\n👉 " + followup, "summary": new_summary}

    # --- CAREER QUERY ---
    resume_data = data.get("resume")
    github_data = data.get("github")
    leetcode_data = data.get("leetcode")
    query = user_input.lower()

    # Depth
    if len(query.split()) <= 5:
        depth = "shallow"
    elif any(w in query for w in ["analyze", "evaluate", "review", "compare"]):
        depth = "deep"
    elif any(w in query for w in ["roadmap", "plan", "improve", "how"]):
        depth = "medium"
    else:
        depth = "medium"

    # Mode
    if any(w in query for w in ["analyze", "evaluate", "review"]):
        mode = "diagnostic"
    elif any(w in query for w in ["improve", "next step", "what should i do"]):
        mode = "prescriptive"
    elif any(w in query for w in ["what is", "explain"]):
        mode = "descriptive"
    else:
        mode = "default"

    # Style
    if depth == "shallow":
        style_instruction = "Answer in 2-3 lines. Be direct."
    elif depth == "medium":
        style_instruction = "Give a clear structured answer."
    else:
        style_instruction = "Give a detailed, analytical response with actionable insights."

    # Cross-analysis (only deep + data present)
    interpretation = None
    if depth == "deep" and (resume_data or github_data or leetcode_data):
        interpretation = llm.invoke(f"""
        {MENTOR_PERSONA}

        Now cross-check this candidate's claims vs proof:

        RESUME (what they CLAIM):
        {resume_data}

        GITHUB (what they actually DID):
        {github_data}

        LEETCODE (DSA proof):
        {leetcode_data}

        Do:
        - Match claims vs proof
        - Identify gaps (be strict but constructive)
        - Identify hidden strengths they might not even realize
        - Consider what Indian recruiters specifically look for
        - Evaluate DSA readiness based on LeetCode stats

        Keep it concise but sharp.
        """).content

    # Format
    if depth == "shallow":
        format_instruction = "Just answer the question. No sections."
    elif depth == "medium":
        format_instruction = """
        Respond in:
        - Direct Answer
        - Key Points
        """
    else:
        format_instruction = """
        Respond in:

        ### ✅ Final Verdict

        ### 🔍 Key Insights

        ### ⚠️ Gaps

        ### 📋 Action Plan
        (Include realistic timelines, free resources, and Indian-market-specific advice)
        """
        if interpretation:
            format_instruction += "\nInclude claim vs proof insights."

    # Final LLM call — prompt instructs model to include a natural follow-up
    final = llm.invoke(f"""
    {MENTOR_PERSONA}

    Conversation Summary:
    {summary}

    User Question:
    {user_input}

    Resume Data:
    {resume_data}

    GitHub Data:
    {github_data}

    LeetCode Data:
    {leetcode_data}

    Cross-Analysis:
    {interpretation}

    Instructions:
    {style_instruction}
    {format_instruction}

    IMPORTANT — Natural Follow-Up:
    After your answer, close with a natural, CONVERSATIONAL follow-up question
    that is directly relevant to what the user asked. This should feel like you're
    genuinely curious about THEIR specific situation so you can help better.
    Examples:
    - If they asked about internships: "By the way, what sector are you targeting —
      product companies, startups, or service companies like TCS/Infosys?"
    - If they asked about DSA: "How comfortable are you with arrays and strings right now?"
    - If they asked about a roadmap: "What's your current strongest skill — do you have
      any projects built yet?"
    Make it feel like a real conversation, not a menu of options.

    Remember:
    - Do NOT repeat raw data back. ANALYZE it.
    - All advice must be grounded in Indian job market realities.
    - Recommend only FREE or affordable resources.
    - If mentioning companies, use Indian companies and CTC ranges.
    - Be the mentor they never had. Be honest, be helpful, show the path.
    """).content

    new_summary = update_summary(summary, user_input, final)
    return {"output": final, "summary": new_summary}
