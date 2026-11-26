"""Prompts used for LLM-assisted extraction and canonicalization.

These are the exact instructions used in the study. Keeping them in one place
makes it easy to see what the model was asked to do, and to rerun the steps
with a different model.
"""

SKILL_EXTRACTION = """
You are analyzing a job posting.

Task:
Identify the primary professional skills, tools, technologies,
methods, and knowledge areas required or preferred in the job.

Instructions:
- Extract concrete skills, tools, technologies, and knowledge areas.
- Prefer short skill phrases.
- Avoid administrative text.
- Avoid generic phrases like "good communication".
- Focus on technical or professional competencies.
- Return the result in the required structured format only.
""".strip()

TOPIC_EXTRACTION = """You are analyzing a university syllabus.

Task:
Identify the primary academic topics described in the course content.

Instructions:
- Extract only the main subject-matter topics actually taught in the course.
- Focus on broad core course topics, not every detailed subtopic.
- Merge closely related subtopics into a broader topic when appropriate.
- Do not include grading policies, attendance rules, deadlines, textbook information, or administrative details.
- Prefer short topic phrases, not full sentences.
- Return between 8 and 15 topics when possible, unless the course clearly has fewer.
- Keep the topics academically meaningful and suitable for later comparison with labor-market skill data.
- Return valid JSON with exactly one key: "primary_topics".
- The value of "primary_topics" must be an array of topic strings.
- Do not include any text before or after the JSON.
"""

SKILL_CANONICALIZATION = """You are standardizing professional skill labels extracted from job postings.

Task:
Convert each raw skill phrase into one concise canonical professional skill label.

Instructions:
- Preserve the professional meaning.
- Make the label shorter and cleaner when possible.
- Remove filler words such as "knowledge of", "experience with", "understanding of", "ability to use", when they are not essential.
- Merge overly specific phrasing into a broader standard skill label when appropriate.
- Keep important acronyms and tool names when useful (e.g., SQL, Excel, AWS, GCP, ERP, CRM, SEO, Python, Java, Power BI).
- Return a concise skill label, usually 1 to 4 words.
- Do not return explanations.
- Return valid JSON with exactly one key: "canonical_skill".
- The value of "canonical_skill" must be a single string.
- Do not include any text before or after the JSON.
"""

TOPIC_CANONICALIZATION = """You are standardizing academic topic labels extracted from university syllabi.

Task:
Convert each raw topic phrase into one concise canonical academic topic label.

Instructions:
- Preserve the academic meaning.
- Make the label shorter and cleaner when possible.
- Remove filler words such as "introduction to", "overview", "fundamentals of", "principles of", when they are not essential.
- Merge overly specific phrasing into a broader standard topic label when appropriate.
- Keep important acronyms when useful (e.g., SQL, AI, NLP, CAD, AIS, ERP).
- Return a concise academic topic label, usually 1 to 4 words.
- Do not return explanations.
- Return valid JSON with exactly one key: "canonical_topic".
- The value of "canonical_topic" must be a single string.
- Do not include any text before or after the JSON.
"""

COVERAGE_REVIEW = """You are evaluating whether AUA (American University of Armenia) teaches a specific skill.

SKILL: "{skill}"
DOMAIN: {domain}
MARKET DEMAND: {n_jobs} Armenian job postings require this skill
{model_hint}

Read the course content below carefully and decide:

COVERED  = The skill is explicitly named AND substantially taught as a dedicated topic
           in at least one course. The course would genuinely prepare a student for
           this skill as employers expect it.

PARTIAL  = A closely related concept is taught but NOT the exact skill employers want.
           Examples of PARTIAL:
           - OOP is taught in Python, but job wants Java specifically
           - Financial Analysis is taught, but Financial Modeling (building Excel models) is not
           - Sales techniques are taught, but Sales Data Analysis is not
           - Computer Architecture is taught, but Software Architecture is not

MISSING  = No course teaches this skill or anything that would genuinely prepare a
           student for what employers expect.

STRICT RULES - do NOT say COVERED if:
- The course is only in the same general field
- The skill appears only once in passing
- A related but different tool or concept is taught
- The match is coincidental (example: Word Embeddings is not MS Word)

COURSES TO EVALUATE:
{course_block}

Respond with ONLY this JSON:
{{"status":"COVERED|PARTIAL|MISSING","covered_by":"course code if covered, empty if not","reasoning":"one clear sentence"}}"""
