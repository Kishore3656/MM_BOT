# Claude.md — ICT Trading Bot

## Token Optimization Rules
- **Default to reading small files only.** Ask before reading large files (>500 lines).
- **Use Grep/Glob for searches** instead of reading entire files.
- **Avoid verbose explanations.** Keep responses direct and actionable.
- **No unnecessary rewrites.** Only modify code when explicitly asked or when fixing bugs.
- **Batch file reads.** Group related reads in single calls when possible.
- **Don't create files unless essential.** Modify existing files or use simple tools instead.

## Project Structure
- `gold_clean_data/` — Cleaned market data (gitignored)
- `nasdaq/`, `backtest_data/` — Raw trading data (gitignored)
- `.venv/` — Python virtual environment
- `bot.log`, `bot_test.log` — Runtime logs
- `trades.csv` — Trade execution history
- `PDF/` — Documentation/references

## Development Guidelines
- **Assumption**: You know what you're building. Be concise and direct.
- **Autonomous mode**: No permission prompts for standard operations (file edits, bash runs). Ask only for destructive/risky actions.
- **Data**: Use cached/local data when available. Don't re-fetch from external sources unnecessarily.
- **Python**: Verify venv is activated before running scripts.

## Common Tasks (Reference)
- Run backtest: `python backtest_script.py`
- Check logs: `tail -f bot.log`
- Review trades: `cat trades.csv`
- Clear data cache: Use gitignored folders only
