# Memory Bank for Charting the Future

This directory contains the **Memory Bank** for the "Charting the Future: Harnessing LLMs for Quantitative Finance" project. The Memory Bank enables coding agents to maintain persistent context and understanding across development sessions, transforming it from a stateless assistant into a knowledgeable development partner.

## What is the Memory Bank?

The Memory Bank is a structured documentation system that allows the coding agent to "remember" project details, architectural decisions, and implementation progress between conversations. When you start a new session, simply ask the agent to "follow your custom instructions in the memory-bank" and it should read these files to rebuild its complete understanding of the project.

## Benefits

### 🧠 **Persistent Context**
- Maintains deep understanding of the multi-agent sector committee architecture
- Preserves knowledge of technical decisions and implementation patterns
- Eliminates need to re-explain project goals and constraints

### 📈 **Consistent Development**
- Ensures predictable interactions aligned with project methodology
- Maintains coding standards and architectural patterns across sessions
- Preserves domain expertise in quantitative finance and LLM agents

### 📚 **Self-Documenting Project**
- Creates valuable project documentation as a natural byproduct
- Provides clear onboarding materials for new team members
- Maintains audit trail of decisions and rationale

### 🔄 **Scalable Knowledge Management**
- Works with projects of any complexity level
- Adapts to evolving requirements and scope changes
- Integrates seamlessly with existing development workflows

## File Structure

This Memory Bank contains six core files organized in a hierarchical structure:

```
memory-bank/
├── projectbrief.md      # Foundation: mission, objectives, scope
├── productContext.md    # Why: problem solving and value creation
├── activeContext.md     # Current: focus, decisions, next steps
├── systemPatterns.md    # How: architecture, patterns, implementations
├── techContext.md       # With: technologies, tools, integrations
└── progress.md          # Status: completed work and roadmap
```

### Core Files Explained

- **projectbrief.md** - The foundation document defining the multi-agent sector committee system for systematic investment decision-making
- **productContext.md** - Explains the tri-pillar methodology and how it creates value for portfolio managers and risk teams
- **activeContext.md** - Current development focus on integrating deep_research_runner.py into the sector-committee package
- **systemPatterns.md** - Architectural patterns including factory design, ETF mappings, and risk frameworks
- **techContext.md** - Technical implementation details covering Python 3.13, OpenAI API, and compliance requirements
- **progress.md** - Comprehensive status tracking and implementation roadmap

## How to Use

### Starting a New Session
```
Ask coding agent to "follow your custom instructions in the memory-bank" at the beginning of any conversation
```

### Updating the Memory Bank
```
Ask coding agent to "update memory bank" after significant progress or changes
```

### Key Commands
- **"follow your custom instructions"** - Loads Memory Bank context
- **"update memory bank"** - Triggers comprehensive documentation review
- **"initialize memory bank"** - Creates initial Memory Bank structure (already done)

## Integration with Quantitative Finance Workflow

This Memory Bank is specifically designed for the sophisticated multi-agent financial analysis system described in the book. It captures:

- **Agent Architecture**: Factory patterns for OpenAI o4-mini and o3-deep-research models
- **ETF Universe**: 11 SPDR sectors with inverse ETF mappings and leverage adjustments
- **Risk Management**: Beta neutralization, concentration limits, and compliance requirements
- **Portfolio Construction**: Score aggregation, tilt mapping, and execution workflows

## Learn More

For complete details on the Memory Bank methodology and best practices, visit:
📖 **[Cline Memory Bank Documentation](https://docs.cline.bot/prompting/cline-memory-bank)**

## Project Context

This Memory Bank supports the companion repository for "Charting the Future: Harnessing LLMs for Quantitative Finance" by Colin Alexander, CFA, CIPM. The project demonstrates practical applications of Large Language Models in systematic investment management while maintaining institutional-grade risk controls and regulatory compliance.

---

*The Memory Bank is Cline's only link to previous work. Its effectiveness depends entirely on maintaining clear, accurate documentation that enables seamless context reconstruction across development sessions.*
