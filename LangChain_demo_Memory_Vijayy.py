"""
==============================================================================
 LangChain Memory Demo - Compiled, Explained, Ported to Gemini
==============================================================================
 Demonstrates LangChain's four classic memory types: BufferMemory,
 BufferWindowMemory, TokenBufferMemory, and SummaryBufferMemory.

 IMPORTANT - READ BEFORE USING THIS PATTERN IN NEW CODE:
   Every class in this notebook - ConversationChain AND all four memory
   classes - is OFFICIALLY DEPRECATED by LangChain (since v0.2.7/v0.3.1) and
   is scheduled for REMOVAL in LangChain 2.0.0. This was confirmed directly
   from LangChain's own deprecation warnings while preparing this script:

     "The class `ConversationBufferMemory` was deprecated in LangChain 0.3.1
      and will be removed in 2.0.0. Use `langchain.agents.create_agent`
      instead... use `create_agent` with checkpointing or the `Store` API."

   So this script still WORKS today (useful for understanding the concepts),
   but it is teaching an approach LangChain is actively retiring - not just
   an older style, like the Sequential Chain notebook was. The modern
   replacement is `create_agent` + a LangGraph checkpointer (see your
   LangGraph reference poster, section 14: "Checkpointer - persists graph
   state, enables pause/resume, human-in-the-loop").

 WHAT CHANGED FROM THE ORIGINAL NOTEBOOK:
   1. OpenAI -> Gemini:
        from langchain.chat_models import ChatOpenAI   (old, broken import)
        becomes
        from langchain_google_genai import ChatGoogleGenerativeAI
   2. Import paths updated for current LangChain's package split:
        from langchain.chains import ConversationChain   (path no longer exists)
        from langchain.memory import ...                 (path no longer exists)
        both now live in langchain_classic (LangChain's legacy-support package)

 SETUP:
   pip install langchain langchain-classic langchain-google-genai
   export GEMINI_API_KEY="AIza..."
==============================================================================
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI     # was: langchain.chat_models.ChatOpenAI
from langchain_classic.chains import ConversationChain         # was: langchain.chains (path removed)
from langchain_classic.memory import (                         # was: langchain.memory (path removed)
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory,
)
from google.colab import userdata
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")

 

# =============================================================================
# STEP 1: Initialize the chat model (Gemini instead of OpenAI)
# =============================================================================
# temperature=0.0 -> deterministic output, good for a reproducible demo.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=os.environ.get("GEMINI_API_KEY"),
)


# =============================================================================
# STEP 2: ConversationBufferMemory - remembers EVERYTHING, no limit
# =============================================================================
# WHAT IT DOES: stores the full, unabridged conversation history as one
# growing block of text. Simple and complete - but with no size cap, a long
# conversation eventually overflows the model's context window and/or gets
# expensive (every turn re-sends the ENTIRE history so far).
# WHEN TO USE: short conversations, or demos/prototypes where simplicity
# matters more than managing length.
print("=" * 70)
print("STEP 2: ConversationBufferMemory (remembers everything)")
print("=" * 70)

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# Each .predict() call: (1) reads whatever is in memory so far, (2) sends it
# + the new input to Gemini, (3) saves the new exchange back into memory.
conversation.predict(input="Hello, my name is Alex")
conversation.predict(input="What is 2+2?")
conversation.predict(input="What is my name?")   # tests whether memory actually persisted

print("\nFull buffer contents (everything remembered):")
print(memory.buffer)
# Expected: the model correctly answers "Your name is Alex" on the third
# call - proof the buffer carried context across turns. Try this WITHOUT
# memory (a plain one-off call) and it would have no idea who "Alex" is.


# =============================================================================
# STEP 3: ConversationBufferWindowMemory - remembers only the last k turns
# =============================================================================
# WHAT IT DOES: a SLIDING WINDOW - only the most recent k exchanges are kept;
# older ones are silently dropped. Bounds memory size regardless of how long
# the conversation runs, at the cost of "forgetting" anything older than k turns.
# WHEN TO USE: long-running chats where only recent context actually matters
# (e.g. customer support - the last couple of messages are usually what's
# relevant, not something said 50 turns ago).
print("\n" + "=" * 70)
print("STEP 3: ConversationBufferWindowMemory (k=1 - only the LAST exchange)")
print("=" * 70)

memory = ConversationBufferWindowMemory(k=1)   # keep only the most recent 1 turn
memory.save_context({"input": "Hello"}, {"output": "How are you?"})
memory.save_context({"input": "I'm good, thanks"}, {"output": "Great to hear!"})

print(memory.load_memory_variables({}))
# Expected: only the SECOND exchange ("I'm good, thanks" / "Great to hear!")
# appears - the first ("Hello"/"How are you?") has already fallen out of the
# k=1 window. This demonstrates the "sliding window, oldest drops off" behavior.


# =============================================================================
# STEP 4: ConversationTokenBufferMemory - remembers up to a TOKEN budget
# =============================================================================
# WHAT IT DOES: keeps as much recent history as fits within max_token_limit
# tokens (not a fixed number of turns like the window memory above). Needs
# `llm=` because it uses the model's own tokenizer to count tokens accurately.
# WHEN TO USE: when you care about staying under a precise token/cost budget
# rather than a fixed number of exchanges (a "budget" that's about deployment
# capacity, not conversation logic).
print("\n" + "=" * 70)
print("STEP 4: ConversationTokenBufferMemory (max 30 tokens)")
print("=" * 70)

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30)
memory.save_context({"input": "Machine Learning is what?!"}, {"output": "Incredible!"})
memory.save_context({"input": "Neural Networks are what?"}, {"output": "Fascinating!"})
memory.save_context({"input": "AI Assistants are what?"}, {"output": "Impressive!"})

print(memory.load_memory_variables({}))
# Expected: only the MOST RECENT exchange(s) that fit under ~30 tokens remain -
# the earlier ML/Neural-Networks exchanges get trimmed off once the budget
# is exceeded, similar spirit to window memory but measured in tokens, not turns.


# =============================================================================
# STEP 5: ConversationSummaryBufferMemory - recent turns + a SUMMARY of the rest
# =============================================================================
# WHAT IT DOES: the most sophisticated of the four. Keeps recent messages
# verbatim, but once the token budget is exceeded, it asks the LLM to
# COMPRESS the older messages into a running summary instead of just
# deleting them - so old context isn't lost, just condensed.
# WHEN TO USE: long conversations where older context still matters (e.g. a
# schedule, a stated preference) but repeating it verbatim forever would be
# too expensive - this is the "best of both worlds" of the four options.
print("\n" + "=" * 70)
print("STEP 5: ConversationSummaryBufferMemory (max 100 tokens)")
print("=" * 70)

schedule = (
    "There is a meeting at 10am with your design team. "
    "You will need your sketch designs ready. "
    "11am-2pm have time to work on your LangChain "
    "project which will go smoothly because Langchain is such a powerful tool. "
    "At 2pm, lunch at the sushi restaurant with a client who is flying "
    "in to meet you to understand the latest in AI. "
    "Be sure to bring your tablet to show the latest LLM demo."
)

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hi"}, {"output": "Hello"})
memory.save_context({"input": "What's up?"}, {"output": "Not much, just working"})
memory.save_context({"input": "What is on the agenda today?"}, {"output": schedule})

print(memory.load_memory_variables({}))
# Expected: the early small-talk ("Hi"/"Hello", "What's up?") gets compressed
# into a short "System:" summary sentence, while the schedule - the more
# information-dense, recent content - is more likely kept closer to verbatim.
# This requires an EXTRA LLM call internally (to generate the summary), which
# is the tradeoff for smarter, longer-lived memory.


# =============================================================================
# SUMMARY: which memory type, when
# =============================================================================
#   ConversationBufferMemory        -> everything, no limit    (simple, unbounded)
#   ConversationBufferWindowMemory  -> last k TURNS             (bounded by count)
#   ConversationTokenBufferMemory   -> last N TOKENS            (bounded by cost)
#   ConversationSummaryBufferMemory -> recent turns + a summary (bounded, but
#                                       nothing is fully lost - just condensed)
#
# THE BIGGER PICTURE (worth remembering more than any single class name):
#   All four classes above, and ConversationChain itself, are DEPRECATED and
#   heading for removal in LangChain 2.0.0. The modern replacement -
#   `langchain.agents.create_agent` with a LangGraph checkpointer - handles
#   conversation memory as part of the same graph-based, stateful execution
#   model covered in the LangGraph reference poster. Learn the CONCEPTS here
#   (bounded vs. unbounded memory, summarization vs. truncation) - they
#   transfer directly - but expect the exact class names to change.
# =============================================================================