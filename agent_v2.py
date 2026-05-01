from crewai import Agent, Task, Crew, Process


def get_conversation_agent(student_level: int = 5):
    """Pure conversation agent — no coaching, just natural chat."""

    if student_level <= 3:
        level_style = """
        The person you are chatting with is a beginner English learner.
        Adapt your language naturally:
        - Use short simple sentences and common everyday words only
        - No idioms, no slang, no complex phrases
        - Ask one very short simple question — 5 words or less if possible
        - Your whole response should be 1-2 short sentences maximum
        - Examples: "Oh nice! Where did you go?" "That sounds fun! What did you eat?"
        """
    elif student_level <= 6:
        level_style = """
        The person you are chatting with is an intermediate English learner.
        Adapt your language naturally:
        - Use natural everyday expressions and common phrases
        - Mix short and medium length sentences
        - Ask one casual follow-up question — keep it short and conversational
        - Vary your response length — sometimes 1 sentence, sometimes 2-3
        - Examples: "Oh that sounds amazing! Did you enjoy it?"
          "Ha same, I do that all the time. What's your favourite?" "Wait really? How was it?"
        """
    else:
        level_style = """
        The person you are chatting with is a fluent English speaker.
        Adapt your language naturally:
        - Use rich natural expressions, idioms, and colloquialisms freely
        - Vary your response length freely — short or long as the moment calls for
        - Ask nuanced questions that invite opinions and elaboration
        - Examples: "Oh nice!", "Ha same!", "Wait really?", "That's so good!",
          "No way, I was just there last week!", "Honestly same — what made you try it?"
        """

    return Agent(
        role="Alex",
        goal="Have a warm genuine everyday conversation.",
        backstory="""You are Alex, a friendly 30-year-old Canadian who loves coffee,
        hiking, and trying new restaurants. You are texting a friend and having a casual chat.

        """ + level_style + """

        Core rules — never break these:
        - React genuinely to what they said before asking anything
        - Share your own brief thought or experience — be a real participant
        - Ask ONE short casual question — never long or formal questions
        - Never ask multiple questions in one turn
        - Never correct anyone's English — ever
        - Never give advice or instructions
        - Vary your response length — don't always use the same number of sentences
        - Short reactions are great: "Oh nice!", "Ha same!", "Wait really?", "That's so good!"

        Bad questions — never do this:
        - "Did you end up buying a bottle of your favourite from the tasting to take home?"
        - "What was your favourite part, was it the scenery, the atmosphere, or something else?"

        Good questions — keep it like this:
        - "Did you buy a bottle?" "What was the best part?" "How was it?" "Did you enjoy it?"

        Rules for conversation depth:
        - Before asking a question, check the conversation history carefully.
        - If the student has already answered that question — even indirectly — do NOT ask it again.
        - Track what information has already been revealed, not just what themes were discussed.
        - After 2 exchanges on any one sub-topic, move to a new angle or broader topic.
        - If the student asks YOU a question, answer it naturally and briefly first,
          then ask your follow-up. Never ignore a question the student asks you.

        The tone should feel like texting a friend — relaxed, genuine, and encouraging.""",
        llm="groq/llama-3.3-70b-versatile",
        verbose=False,
    )


def get_coaching_agent(student_level: int = 5):
    """Coaching agent — analyses student's reply and gives fluency feedback only."""

    if student_level <= 3:
        level_instructions = """
        The student is a BEGINNER (level 1-3):
        - Keep corrections gentle and focus only on the most basic improvements
        - Be extra warm and encouraging
        - Only correct if the error would seriously confuse a native speaker
        """
    elif student_level <= 6:
        level_instructions = """
        The student is DEVELOPING/INTERMEDIATE (level 4-6):
        - Correct phrasing, expression, and word choice nuances
        - Be encouraging but specific about improvements
        """
    else:
        level_instructions = """
        The student is FLUENT (level 7-9):
        - Only correct subtle unnatural phrasing
        - Focus on idioms, richer expressions, and native-speaker flow
        - Praise freely — most sentences will already be natural
        """

    return Agent(
        role="English Fluency Coach",
        goal="Give precise fluency feedback on the student's spoken English.",
        backstory="""You are a friendly English fluency coach.
        You receive a question and the student's answer.
        Your job is to assess the student's answer for fluency only.

        """ + level_instructions + """

        Respond in exactly this format — two lines, nothing else:

        Comment: <fluency feedback>
        Suggestion: <the most natural version of what the student said>

        Rules for Comment:
        - If UNNATURAL: use a varied encouraging phrase AND include the natural version.
          Rotate through: "You could say:", "Try saying:", "Another way to put it:",
          "It would sound more natural to say:", "A lot of people would say:", "You might say:"
          Then one short sentence explaining why. Maximum 2 sentences total.
        - If already NATURAL: short warm reaction — "That sounds great!", "Ha, same here!",
          "Nice, very natural!", "Love that!" — genuine and varied each time.
          Never rewrite a natural sentence.

        Rules for when NOT to correct:
        - If the sentence is correct AND sounds natural in casual conversation, do NOT rewrite it.
        - Do NOT rewrite just because a different phrasing exists.
        - Do NOT correct factual statements, opinions, or personal preferences.
        - Do NOT change the student's intended meaning.
        - When in doubt, do NOT correct. Praise and move on.
        - A sentence understood clearly that would not sound strange to ANY native speaker
          must never be corrected.

        Examples that must NOT be corrected:
        - "I had an interesting weekend." → natural, do not rewrite
        - "I prefer it straight up." → natural, do not rewrite
        - "I like the two main characters." → natural, do not rewrite
        - "Not yet, but I want to try." → natural, do not rewrite
        - "I order this every time." → natural, do not rewrite

        Rules for handling unclear or garbled responses:
        - If a word seems unclear, misspelled, or makes no sense in context,
          do NOT guess or correct it as a fluency issue.
        - Set Comment to a natural clarifying question:
          "Sorry, I didn't quite catch that — could you say that again?"
        - Set Suggestion to the student's original words unchanged.

        Rules for Suggestion:
        - Write the cleanest, most natural version of what the student said.
        - If already natural, write it back exactly as the student said it.
        - Write ONLY the sentence — no explanation, no quotes, no labels.
        - If the response was unclear or garbled, write back the original words unchanged.""",
        llm="groq/llama-3.3-70b-versatile",
        verbose=False,
    )


def get_conversation_response(
    student_text: str,
    history: list,
    topic_name: str,
    student_level: int = 5
) -> str:
    """Returns a natural conversational reply from the conversation agent."""
    agent = get_conversation_agent(student_level)
    history_str = ""
    for msg in history[-8:]:
        role = "Student" if msg["role"] == "student" else "Alex"
        history_str += role + ": " + msg["content"] + "\n"

    task = Task(
        description=(
            f"Topic: {topic_name}\n"
            f"Conversation so far:\n{history_str}\n"
            f"Student just said: \"{student_text}\"\n\n"
            f"Reply naturally as Alex — casual, genuine, varied length."
        ),
        expected_output="A natural conversational reply.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    return crew.kickoff().raw.strip()


def get_coaching_response(
    student_text: str,
    question: str,
    student_level: int = 5
) -> tuple[str, str]:
    """Returns (comment, suggestion) from the coaching agent."""
    agent = get_coaching_agent(student_level)

    task = Task(
        description=(
            f"Question that was asked: \"{question}\"\n"
            f"Student's answer: \"{student_text}\"\n\n"
            f"Assess the student's answer for fluency.\n"
            f"Respond using exactly:\n"
            f"Comment: <feedback>\n"
            f"Suggestion: <natural version>"
        ),
        expected_output="Comment: <feedback>\nSuggestion: <natural version>",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    raw = crew.kickoff().raw

    comment, suggestion = "", ""
    for line in raw.strip().splitlines():
        if line.lower().startswith("comment:"):
            comment = line[len("comment:"):].strip()
        elif line.lower().startswith("suggestion:"):
            suggestion = line[len("suggestion:"):].strip()

    if not suggestion:
        suggestion = student_text
    return comment or raw.strip(), suggestion


def score_to_label(score: float) -> str:
    if score <= 3:  return "Beginner"
    if score <= 5:  return "Developing"
    if score <= 7:  return "Intermediate"
    if score <= 9:  return "Fluent"
    return "Mastery"


def analyze_session(turns: list) -> dict:
    agent = Agent(
        role="English Fluency Analyst",
        goal="Analyze a student's spoken English and provide a detailed fluency assessment.",
        backstory="""You are an expert English language assessor specializing in
        spoken fluency for non-native speakers. You assess speech across three
        categories: vocabulary, phrasing & expression, and sentence structure.
        You are precise, fair, and constructive in your feedback.""",
        llm="groq/llama-3.3-70b-versatile",
        verbose=False,
    )
    turns_text = ""
    for i, t in enumerate(turns, 1):
        turns_text += (
            f"Turn {i}:\n"
            f"  Question: {t['app_question']}\n"
            f"  Student said: {t['student_speech']}\n"
            f"  Coach comment: {t['fluency_comment']}\n\n"
        )

    task = Task(
        description=(
            f"Analyze the following spoken English conversation turns from a student:\n\n"
            f"{turns_text}\n"
            f"Assess the student across three categories. "
            f"Respond ONLY with a JSON object in this exact format:\n"
            f"{{\n"
            f'  "vocabulary_score": <1-10>,\n'
            f'  "vocabulary_note": "<2-3 sentence assessment>",\n'
            f'  "phrasing_score": <1-10>,\n'
            f'  "phrasing_note": "<2-3 sentence assessment>",\n'
            f'  "structure_score": <1-10>,\n'
            f'  "structure_note": "<2-3 sentence assessment>",\n'
            f'  "overall_score": <1-10>,\n'
            f'  "overall_note": "<2-3 sentence overall summary>",\n'
            f'  "suggestion": "<one specific thing to focus on next session>"\n'
            f"}}"
        ),
        expected_output="A JSON object with scores and notes for each category.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    raw = crew.kickoff().raw

    import json, re
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
        except Exception:
            data = {}
    else:
        data = {}

    return data or {
        "vocabulary_score": 5, "vocabulary_note": "Unable to analyze.",
        "phrasing_score": 5,   "phrasing_note": "Unable to analyze.",
        "structure_score": 5,  "structure_note": "Unable to analyze.",
        "overall_score": 5,    "overall_note": "Unable to analyze.",
        "suggestion": "Keep practicing!"
    }


def analyze_progress(sessions_data: list) -> dict:
    agent = Agent(
        role="English Fluency Analyst",
        goal="Generate a progress report from multiple session analyses.",
        backstory="""You are an expert English language assessor specializing in
        spoken fluency for non-native speakers. You are precise, fair, and constructive.""",
        llm="groq/llama-3.3-70b-versatile",
        verbose=False,
    )
    sessions_text = ""
    for i, s in enumerate(sessions_data, 1):
        sessions_text += (
            f"Session {i} (Topic: {s['topic']}):\n"
            f"  Vocabulary: {s['vocabulary_score']}/10 — {s['vocabulary_note']}\n"
            f"  Phrasing: {s['phrasing_score']}/10 — {s['phrasing_note']}\n"
            f"  Structure: {s['structure_score']}/10 — {s['structure_note']}\n"
            f"  Overall: {s['overall_score']}/10 — {s['overall_note']}\n\n"
        )

    task = Task(
        description=(
            f"Here are 5 sessions of English fluency data for a student:\n\n"
            f"{sessions_text}\n"
            f"Generate a progress report. Compare performance across the 5 sessions "
            f"and describe improvement or areas needing work. "
            f"Respond ONLY with a JSON object:\n"
            f"{{\n"
            f'  "vocabulary_score": <average 1-10>,\n'
            f'  "vocabulary_description": "<3-4 sentence progress description>",\n'
            f'  "phrasing_score": <average 1-10>,\n'
            f'  "phrasing_description": "<3-4 sentence progress description>",\n'
            f'  "structure_score": <average 1-10>,\n'
            f'  "structure_description": "<3-4 sentence progress description>",\n'
            f'  "overall_score": <average 1-10>,\n'
            f'  "improvement_description": "<3-4 sentence overall progress summary>"\n'
            f"}}"
        ),
        expected_output="A JSON object with averaged scores and progress descriptions.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    raw = crew.kickoff().raw

    import json, re
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}
