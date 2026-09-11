# Dynamic Human–Agent Affective Cooperation

**Research prototype for studying dynamic human–LLM cooperation through interpretable, temporally persistent interaction state.**

This prototype investigates whether an AI agent can cooperate more appropriately with a human when it maintains a **bounded and interpretable interaction state across turns**, rather than responding only to the current message.

The central research question is:

> **Does temporally persistent affect-sensitive state improve human–AI cooperation compared with task-focused and current-cue-only agents?**

## Prototype Overview

The system separates **human/task observations**, **estimated human and interaction state**, **agent state**, **policy selection**, and **response generation** so that changes in agent behaviour remain explicit and inspectable.

At each interaction turn:

**Human input + task context**
→ **Human-state appraisal**
→ **Goal and interaction-state representation**
→ **Agent target state**
→ **Temporal state update**
→ **Cooperation policy**
→ **LLM response**

The agent can select cooperation policies including:

* **INFORM** — provide relevant task information
* **CLARIFY** — resolve uncertainty, ambiguity, or conflict
* **ACKNOWLEDGE** — recognise relevant affective cues
* **REDIRECT** — adapt the interaction when stronger intervention is appropriate
* **DEFER** — respect safety, autonomy, or uncertainty constraints

The response generator does **not receive the raw numerical agent state directly**. Agent state influences behaviour through an explicit policy layer, helping keep the mechanism interpretable.

## Experimental Conditions

The prototype supports three controlled conditions:

| Condition        | Affective appraisal            | Persistent agent state | Purpose                                        |
| ---------------- | ------------------------------ | ---------------------- | ---------------------------------------------- |
| **TASK_FOCUSED** | Logged but not used for policy | No                     | Task-oriented baseline                         |
| **CURRENT_CUE**  | Yes                            | No (`ρ = 0`)           | Responds to the current interaction state only |
| **DYNAMIC**      | Yes                            | Yes                    | Maintains interaction state across turns       |

This design allows the prototype to examine whether **temporal persistence itself** changes cooperative behaviour while keeping the underlying task and interaction context comparable.

## How the Dynamic Condition Works

For affect-enabled conditions, the system estimates a target agent state from the current interaction.

The agent state represents three functional priorities:

* **Motivational priority** — how strongly the situation relates to important human goals.
* **Decision-information priority** — how strongly the interaction requires clarification or additional evidence.
* **Intervention readiness** — whether stronger adaptive support may be appropriate.

In the **DYNAMIC** condition, the current agent state combines the previous state with the newly estimated target state:

`A_t = ρA_(t−1) + (1−ρ)A*_t`

This allows interaction history to influence the agent gradually rather than allowing one message to completely redefine its behaviour.

In **CURRENT_CUE**, `ρ = 0`, so only the current-turn target state is used.

## What the Prototype Demonstrates

The prototype allows a researcher to:

* interact with the agent across multiple turns;
* compare **TASK_FOCUSED**, **CURRENT_CUE**, and **DYNAMIC** conditions;
* inspect estimated human and interaction variables;
* observe how agent state changes across turns;
* see which cooperation policy was selected;
* compare responses generated from the same interaction context;
* replay turns for controlled comparison; and
* inspect turn-level records for later analysis.

## Technology

* **Python 3.11+**
* **Streamlit** — interactive research interface
* **Pydantic** — validated state representations
* **Provider-agnostic LLM adapter**
* **YAML configuration**
* **JSONL / SQLite-compatible logging**
* **pytest** — implementation and condition-isolation tests

## Running the Prototype

```bash
git clone https://github.com/ijazulhaq1/dynamic-human-agent-affective-cooperation.git

cd dynamic-human-agent-affective-cooperation

pip install -r requirements.txt

streamlit run prototype/app.py
```

LLM credentials, when required, should be provided through environment variables and are **not stored in the repository**.

## Repository Structure

```text
prototype/
├── app.py
├── models/       # human, goal, interaction and agent states
├── services/     # appraisal, transition, policy and response pipeline
├── llm/          # model adapter and prompting
├── ui/           # interaction and researcher dashboards
├── config/       # experimental parameters and conditions
└── tests/        # validation and experimental-isolation tests
```

For additional technical detail, see the accompanying **Prototype Implementation Overview**.

## Research Status

This repository is a **research prototype** designed to make the proposed mechanism concrete, inspectable, and experimentally testable.

It does not assume that an LLM directly observes a person's internal emotional state. Human-state variables are treated as **bounded estimates based on observable interaction cues**, accompanied by confidence information and constrained by explicit safety and autonomy rules.

---

**Ijaz Ul Haq**
Researcher in Artificial Intelligence, Human–AI Interaction, and Technology-Enhanced Learning
