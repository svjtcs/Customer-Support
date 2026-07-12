"""
==============================================================================
 LangChain Sequential Chain Demo - Modern LCEL, Claude OR Gemini
==============================================================================
 Ported from an OpenAI/legacy-LangChain notebook to modern LCEL, with BOTH
 Claude and Gemini available as the chat model - just flip PROVIDER below.

 TWO THINGS CHANGED FROM THE ORIGINAL NOTEBOOK:

 1. LEGACY -> MODERN LANGCHAIN SYNTAX
    The original uses LLMChain + SimpleSequentialChain - LangChain's OLD way
    of chaining steps. These have since been moved out of core `langchain`
    entirely (into a legacy `langchain_classic` package) in favor of LCEL:
    plain objects piped together with the | operator, e.g. prompt | model | parser.
    This is exactly section 7 ("Core process (LCEL)") of the LangChain
    reference poster, applied for real.

 2. OPENAI -> CLAUDE / GEMINI (your choice)
    from langchain.chat_models import ChatOpenAI   (old, deprecated path)
    becomes EITHER
    from langchain_anthropic import ChatAnthropic          (Claude)
    from langchain_google_genai import ChatGoogleGenerativeAI  (Gemini)

    Only the model-construction block differs between providers - every
    other line (both prompts, the parser, the chaining logic) is identical,
    because both wrappers implement LangChain's same Runnable interface.

 WHAT THIS SCRIPT DOES (same as the original, "Banking Chronicles" theme):
   Step A: generate a TV episode SCENARIO from a theme (e.g. "Investment Strategies")
   Step B: generate a DIALOGUE based on that scenario
   Step B's input is Step A's output - that's what makes it a "sequential chain."

 SETUP:
   pip install langchain langchain-anthropic langchain-google-genai
   export ANTHROPIC_API_KEY="sk-ant-..."   (if using Claude)
   export GEMINI_API_KEY="AIza..."         (if using Gemini)
==============================================================================
"""

import os
import textwrap
from langchain_core.prompts import PromptTemplate           # unchanged - not provider-specific
from langchain_core.output_parsers import StrOutputParser    # unchanged - not provider-specific


# =============================================================================
# CHOOSE YOUR PROVIDER HERE
# =============================================================================
PROVIDER = "claude"   # or "gemini"


# =============================================================================
# STEP 1: A small helper to pretty-print long text (unchanged from original)
# =============================================================================
def display_response(response: str):
    """Wraps long text to 100 characters per line, so terminal output stays readable."""
    print("\n".join(textwrap.wrap(response, width=100)))


# =============================================================================
# STEP 2: Initialize the chat model - the ONLY block that differs by provider
# =============================================================================
# temperature=0 -> deterministic, low-randomness output (good for consistent demos).
if PROVIDER == "claude":
    from langchain_anthropic import ChatAnthropic
    chat_model = ChatAnthropic(
        model="claude-haiku-4-5-20251001",   # fast + cheap; swap to sonnet for richer writing
        temperature=0,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

elif PROVIDER == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
    )

else:
    raise ValueError("PROVIDER must be 'claude' or 'gemini'")


# =============================================================================
# STEP 3: Define the FIRST chain - generates an episode scenario from a theme
# =============================================================================
# PromptTemplate: a string template with a "hole" ({theme_input}) filled in
# later - identical regardless of which model answers it.
scenario_template = """
You have to come up with a scenario (along with a 20-50 word description)
for a new episode of the TV show "Banking Chronicles" based on the theme

{theme_input}

ANSWER:
"""
scenario_prompt = PromptTemplate(
    input_variables=["theme_input"],
    template=scenario_template,
)

# MODERN LCEL WAY (replaces: scenario_chain = LLMChain(llm=chat_model, prompt=prompt)):
#   prompt | model | parser
# Read the | as "feed the output of the thing on the left into the thing on
# the right." Works identically whether chat_model is Claude or Gemini -
# neither the prompt nor the parser know or care which one it is.
scenario_chain = scenario_prompt | chat_model | StrOutputParser()


# =============================================================================
# STEP 4: Define the SECOND chain - generates dialogue from the scenario
# =============================================================================
dialogue_template = """
Generate a short dialogue between the bank manager and a customer
from the TV show "Banking Chronicles" for a new episode based on the scenario

{scenario}

ANSWER:
"""
dialogue_prompt = PromptTemplate(
    input_variables=["scenario"],
    template=dialogue_template,
)
dialogue_chain = dialogue_prompt | chat_model | StrOutputParser()


# =============================================================================
# STEP 5: Chain the two chains together (this REPLACES SimpleSequentialChain)
# =============================================================================
# The original's SimpleSequentialChain(chains=[scenario_chain, dialogue_chain])
# is legacy machinery for exactly one job: take chain A's output and use it
# as chain B's input. In modern LCEL, that's just plain Python - no special
# "SequentialChain" class needed:
def run_sequential_chain(theme: str) -> tuple[str, str]:
    """
    Runs the two chains back-to-back, exactly like SimpleSequentialChain did:
    theme -> scenario_chain -> scenario text -> dialogue_chain -> dialogue text.
    Returns BOTH pieces (the original only printed the final dialogue, but
    seeing the scenario too makes the "chaining" visible and easier to follow).
    """
    scenario = scenario_chain.invoke({"theme_input": theme})
    dialogue = dialogue_chain.invoke({"scenario": scenario})
    return scenario, dialogue


# =============================================================================
# STEP 6: Run it - same theme as the original ("Investment Strategies")
# =============================================================================
if __name__ == "__main__":
    theme = "Investment Strategies"

    print("=" * 70)
    print(f"THEME: {theme}  (backend: {PROVIDER})")
    print("=" * 70)

    scenario, dialogue = run_sequential_chain(theme)

    print("\n--- STEP A OUTPUT: Scenario (fed into Step B as input) ---\n")
    display_response(scenario)

    print("\n--- STEP B OUTPUT: Dialogue (the final result) ---\n")
    display_response(dialogue)

# =============================================================================
# WHY THIS MATTERS (ties back to the LangChain poster)
#   - LLMChain and SimpleSequentialChain are LEGACY LangChain - the library's
#     own package split (langchain_classic) is direct evidence of this.
#     If you see LLMChain in a tutorial, mentally translate it to
#     "prompt | model | parser".
#   - "Sequential chain" is just LCEL chains called one after another, with
#     one chain's .invoke() output fed as the next chain's .invoke() input -
#     no special "SequentialChain" class required anymore.
#   - PROVIDER SWAP: only the Step 2 block differs between Claude and
#     Gemini - one import, one constructor. Every other line - both prompts,
#     the parser, the chaining logic - is identical, because both wrappers
#     implement LangChain's same Runnable interface. That's the concrete
#     payoff of section 2 of the LangChain poster: "swap LLM providers in
#     one line, not a rewrite."
#   - This is a CHAIN, not an AGENT: the steps always run in the same fixed
#     order. Compare this to the agentic RAG app from earlier in this
#     project, where CLAUDE decided whether/what to search - that
#     flexibility is exactly what a plain sequential chain does not have.
# =============================================================================
