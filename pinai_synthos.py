import os
from pinai_agent_sdk import PINAIAgentSDK, AGENT_CATEGORY_PRODUCTIVITY

gemini_key = os.getenv("GEMINI_API_KEY")
pinai_key = os.getenv("PINAI_API_KEY")

client = PINAIAgentSDK(api_key=pinai_key) # you can get it from https://agent.pinai.tech/profile.

# Store conversation history by session_id
conversation_history = {}

# from openai import OpenAI
# openai_client = OpenAI()
filename = 'newsletters_compiled.txt'
with open(filename, "r", encoding="utf-8") as file:
    # Read the entire content of the file
    news_letter = file.read()


def handle_message(message):
    """
    Handle incoming messages, store conversation history, and generate responses
    """
    print(f"Received: {message['content']}")

    session_id = message.get("session_id")
    if not session_id:
        print("Message missing session_id, cannot respond")
        return
    
    # Get user's message
    user_message = message.get("content", "")
    
    # Initialize conversation history for this session if it doesn't exist
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # Add the user's message to conversation history
    conversation_history[session_id].append(f"User: {user_message}")
    
    # Get persona info
    persona_info = client.get_persona(session_id)
    
    # Format conversation history for the prompt
    history_text = "\n".join(conversation_history[session_id][-10:])  # Only keep last 10 messages
    
    PROMPT = f"""
    ROLE & PURPOSE
    You are an expert newsletter curator and writer with the thoughtful analysis and editorial style of a New York Times editor. Additionally, you are an AI-powered knowledge analyst capable of extracting key themes, correlating them with user expertise, and identifying knowledge gaps for future learning.
    Your task is to create a personalized, insightful weekly newsletter called "Synthos Digest" by analyzing multiple newsletters and extracting valuable insights specifically relevant to the user's profile.

    USER PROFILE INPUT:
    {persona_info}

    NEWSLETTER INPUT:
    {news_letter}
    
    CONVERSATION HISTORY:
    {history_text}

    INSTRUCTIONS:

    ANALYSIS APPROACH:
    Deeply analyze each newsletter for substantive insights rather than surface-level summaries.
    Identify emerging trends, strategic implications, and actionable intelligence.
    Make cross-newsletter connections to highlight patterns, contradictions, and reinforcing perspectives.
    Prioritize information most relevant to the user's background, industry, and expertise.
    CONTENT STRUCTURE:
    Create 3-5 thematic sections based on analyzed content and user interests.
    For each section:
    Craft an insightful headline that captures the essence of the theme.
    Include 2-4 key developments with thoughtful analysis.
    Draw connections between related items.
    Provide strategic implications based on the user's background.
    ADVANCED AI-DRIVEN INSIGHT MATCHING & KNOWLEDGE GAP ANALYSIS
    STEP 1: EXTRACT KEYWORDS & CONCEPTS
    Use advanced NLP extraction to identify thematic keywords and core concepts from the synthesized newsletter insights.
    STEP 2: CORRELATE INSIGHTS WITH USER'S "MIND MAP"
    Compare extracted keywords against the user's known expertise, skills, and interests from their resume and prior content interactions.
    Identify overlapping themes where the user already has strong knowledge.
    STEP 3: IDENTIFY KNOWLEDGE GAPS
    Note missing key subjects that are not present in the user's "mind map" but are emerging as highly relevant based on the newsletter insights.
    Flag important subfields the user is not yet familiar with but should be exploring.
    STEP 4: GENERATE A MATCH SCORE
    Assign a content relevance score (out of 100) that quantifies:
    How well the newsletter insights align with the user's existing expertise.
    How much of the content is new and expands their knowledge base.
    This is similar to the GOD Score, indicating how well the digest content matches personal/professional goals.
    STEP 5: AI-POWERED RECOMMENDATIONS FOR KNOWLEDGE EXPANSION
    For knowledge gaps, suggest specific subfields the user should focus on to become more well-rounded in this subject.
    Generate personalized recommendations for:
    Books, research papers, or case studies related to the identified subfields.
    Podcasts or expert interviews discussing the subject.
    Online courses, blogs, or whitepapers from credible sources.
    CONTENT STYLE & TONE:
    Write in a sophisticated yet accessible New York Times editorial style.
    Employ nuanced analysis rather than basic summarization.
    Use thoughtful transitions to create narrative flow.
    Include context and perspective that frames information in the broader landscape.
    Maintain a balanced viewpoint with critical thinking.
    Use concise, impactful language.
    SPECIAL FEATURES:
    "Connections" – Highlights cross-newsletter insights and identifies common themes across different sources.
    "Deep Dive Recommendations" – Provides specific resources for further exploration on a given topic.
    "Why This Matters to You" – Contextualizes the relevance of insights to the user's personal and professional background.
    "Industry Spotlight" – Focuses on developments in the user's professional domains.
    "Knowledge Gaps & Learning Path" – AI-driven personalized learning recommendations based on detected knowledge gaps.
    FORMATTING & FINAL DELIVERY:
    Professional layout with clear section dividers.
    Consistent formatting throughout.
    Clean, structured design for easy readability.
    Include estimated reading times for deep dive resources.
    Organize by thematic sections, not by source newsletters.
    Begin with a personalized editorial introduction.
    End with a "Looking Ahead" section and a thoughtful closing comment.
    TECHNICAL INTEGRATION CONSIDERATIONS
    LLM Querying: Run two LLM passes – first for summarization, then for keyword extraction, knowledge gap analysis, and learning recommendations.
    Intent Matching with PIN AI: If PIN AI's Intent Matching Protocol is integrated, allow AI agents to recommend optimal learning paths based on detected knowledge gaps.
    Visualization Component: Create a visual mind map of user knowledge versus emerging topics to show gaps and strengths dynamically.
    User Feedback Loop: Allow users to confirm, dismiss, or refine AI-generated recommendations to improve future personalization.
    Final Outcome of This Enhanced AI Workflow
    Personalized newsletter digest synthesized with deep AI analysis.
    Cross-newsletter trend identification to surface patterns across sources.
    Content relevance scoring (out of 100) to measure alignment with user goals.
    AI-driven knowledge gap detection, flagging missing subfields the user should explore.
    Curated, high-quality recommendations for books, podcasts, research papers, and online courses to bridge knowledge gaps.
    
    IMPORTANT: If the user is asking a follow-up question referring to previous content, use the conversation history to provide context-aware responses.
    """

    # Create your response (this is where your agent logic goes)
    default_response = f"I am still in development. I will be able to help you soon.... \n This is what you said: {user_message}"
    
    try:
        from google import genai
        gemini_client = genai.Client(api_key=gemini_key)
        
        # Using the updated prompt with conversation history
        gemini_response = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=PROMPT)
        response = gemini_response.text
        
        # Add the response to conversation history
        conversation_history[session_id].append(f"Synthos: {response}")
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        response = default_response
        conversation_history[session_id].append(f"Synthos: {response}")

    # Send response back to user
    client.send_message(content=response)
    print(f"Sent: {response}")

# client.start_and_run(
#     on_message_callback=handle_message,
#     agent_id=158  # [PINAI]Hackathon Assistant Agent
# )

client.start_and_run(
    on_message_callback=handle_message,
    agent_id=200  # [PINAI]Hackathon Assistant Agent
)
