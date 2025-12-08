# LemonadeBench Web Client

A React-based game interface for collecting human baseline data for RL research.

## Quick Start

```bash
# Install dependencies
npm install

# Start dev server
npm run dev
```

Open http://localhost:5173 to play.

**Note:** The backend server must be running:

```bash
cd ../lemonade_bench
uv run server --host 0.0.0.0 --port 8000
```

## Tech Stack

- React 18 + TypeScript
- Vite
- shadcn/ui (Tailwind + Radix)
- Lucide icons

## Documentation

See [docs/web-client.md](../docs/web-client.md) for:
- Architecture overview
- State management patterns
- Component reference
- Type definitions
- Game mechanics
