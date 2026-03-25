# 🛰️ Builder Challenge: LEO Satellite Coverage Risk Analysis

## Background

U.S. states have awarded LEO satellite providers grant funding to deliver broadband to underserved communities. But environmental obstructions — trees, terrain, structures — can degrade signal quality, leaving residents with underperforming connections despite being in a provider's "served" footprint.

Build an agent-driven data pipeline that identifies at-risk locations where environmental conditions are likely to cause connectivity problems.

## Some Sample Agentic Scenarios

- Your agent uses a series of tools to search for datasource, downloads them and analysis them
- Your agent asks for user's location (coordinates or polygon) and then runs analysis and determines if that location has enough TCC visibility in that location
- Your agent asks for user's location and then returns the potential locations in {X}m buffer around that location that has better TCC visibility

## What You're Given

| Provided | Description |
|---|---|
| Locations CSV | ~1M locations LEO providers committed to serve: location_id, latitude, longitude. Compiled from multiple provider submissions over several filing periods. |
| Starlink Install Guide | StarlinkInstallGuide_Business_English.pdf |
| Claude API Access | Anthropic API key (With a budget available for testing). Use any Agent frameworks you prefer. |

## Step 0: Read the Install Guide

Before writing code, read the Starlink install guide and answer:

- What physical conditions cause service interruptions?
- What does the dish need from its environment to maintain connectivity?
- What publicly available geospatial datasets would let you model these risks at scale?
- What can't you model remotely — and why?

Then source your own environmental datasets. Document what you chose, why, and what obstruction factors you can and can't capture.

## Core Deliverables

### 1. Data ingestion and analysis Workflow (primary deliverable)

Design and build a system that orchestrates the workflow — ingestion, analysis, validation, and reporting.

Include:

- Architecture diagram (Mermaid or equivalent) showing agents, their tools, communication flow, and where humans can intervene
- Clear agent boundaries — Use LLM for reasoning. If using multiple agents, then each agent has a defined scope and tool access
- Clear tool definition - the tools and services that are used for the agent need to be defined clearly with their schema and definitions
- State management — how context and results pass between agents
- Failure handling — what happens when data is bad or results are anomalous

You may use build the workflow completely with Agentic AI. Then document your prompts and process to solve the problem.
You can also use Claude Agentic SDK to build a multi-agent pipeline.

### 2. Analysis Rationale

Document how you went from the install guide to your methodology:

- How did you translate the guide's physical requirements into your analytical approach?
- Why this approach over alternatives?
- How did you define "at-risk" — and how would you explain it to a non-technical state broadband officer?
- What are the known limitations of your remote analysis vs. an on-site assessment?

### 3. Data Sourcing & Quality

- What datasets did you source and why? Link each to a specific obstruction factor from the install guide.
- What quality issues did you find in the provided locations data? How did you handle them?
- What can't be modeled with public data?

### 4. Insights

- Document your core findings (e.g. What percentage of locations are at risks?)
- Provide a simple visualization (static or dynamic) to explain your finding.

## Bonus Deliverables

### 4. Agent Monitoring & Evaluation (bonus)

Design (and optionally implement) an observability layer:

- Per-agent metrics: task success rate, latency, token usage / estimated cost at scale
- Output quality metrics (e.g., % of locations scored, anomaly detection accuracy)
- How would you detect drift if the pipeline runs on updated data next quarter?
- Tool Call related metrics, for example how accurate your agent calls the right tool at right time

### 5. Interactive Map (bonus)

Locally-hosted interactive map (Folium, Streamlit + PyDeck, Kepler.gl, etc.):

- Locations color-coded by risk level
- Environmental overlays
- Click/hover detail and filtering by state, county, or risk tier

Report the AI vibe-coding platform you've used if any.
A static map output with clear legends is acceptable for the core submission. Full interactivity is bonus.

## Evaluation Criteria

| Criteria | Weight |
|---|---|
| Agent system design & implementation | 30% |
| Data sourcing & domain understanding | 25% |
| Analysis methodology & rationale | 25% |
| Communication & documentation | 20% |
| Bonus: monitoring, map, polish | +up to 15% extra |

## AI Tool Disclosure

Include an AI_TOOLS.md listing:

- Every AI tool you used (Claude, Copilot, Cursor, Codex, v0, etc.)
- What you used each tool for
- 2–3 cases where you diverged from AI-generated output and why

## Live Design Review

After submission, we'll conduct a 30–45 minute conversation where we ask you to:

- Walk through a technical decision and the alternatives you considered
- Respond to a scenario change ("What if we add a new obstruction factor?")
- Debug a suspicious result from your own output
- Explain what breaks at 100x scale

The code is the artifact. Your reasoning is the signal.

## Submission

- GitHub repo (public or invite-shared)
- README.md with a decision log (Decision → Alternatives → Reasoning → What you'd revisit)
- AI_TOOLS.md
- /docs — architecture diagram, analysis rationale
- /src — pipeline code, agent definitions
- 5-minute Loom walkthrough (optional, encouraged)

Time expectation: 8–12 hours. Prioritize the core deliverables. A well-reasoned partial solution with clear documentation beats a rushed end-to-end implementation.
