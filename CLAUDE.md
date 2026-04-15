# Claude Code Configuration - RAVE-TFG

## Behavioral Rules

- Do what has been asked; nothing more, nothing less
- NEVER create files unless they're absolutely necessary for achieving your goal
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested
- NEVER save working files, text/mds, or tests to the root folder
- ALWAYS read a file before editing it
- NEVER commit secrets, credentials, or .env files

## File Organization

- NEVER save to root folder — use the directories below
- Use `/src` for source code files
- Use `/tests` for test files
- Use `/docs` for documentation and markdown files
- Use `/tools` for utility scripts

## Project Architecture

- Keep files under 350 lines (target 250)
- Ensure input validation at system boundaries
- This is a Python project — no npm/node tooling

## Security Rules

- NEVER hardcode API keys, secrets, or credentials in source files
- NEVER commit .env files or any file containing secrets
- Always validate user input at system boundaries

## TFG Writing Rules

These rules apply whenever writing or editing TFG document sections.

- Write first in Spanish (Catalan/Spanish) for review; translate to English only after approval
- Every factual claim must be backed by a verifiable external source (academic paper, official doc, etc.)
- Provide all citation links in the text and in a references list at the end of each section
- NEVER assume technical details about the project — if uncertain, ask before writing
- Do not describe what the code "could do"; describe only what it actually does (verify in source)
- Use present tense as primary verb tense; perfect tense is acceptable for completed work
- Do not use passive voice
- Each section must follow: Context/motivation → Objective → Methodology → Results/contributions
- Keep citations in the format: [N] Author(s). (Year). Title. Source. URL
- Approximately 250 words for the abstract; section length proportional to complexity
