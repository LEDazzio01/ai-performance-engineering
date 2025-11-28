# AI Performance Dashboard (Next.js)

A modern, high-performance dashboard for GPU benchmark visualization and optimization insights.

## Quick Start

### Prerequisites

- Node.js 18+ 
- The Python backend server running on port 6970

### Installation

```bash
cd tools/dashboard-next
npm install
```

### Development

1. Start the Python backend server (in another terminal):
```bash
python -m tools.dashboard.server --port 6970
```

2. Start the Next.js development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000)

### Production Build

```bash
npm run build
npm start
```

## Architecture

```
src/
├── app/                 # Next.js App Router
│   ├── layout.tsx       # Root layout with fonts and background
│   ├── page.tsx         # Main dashboard page
│   └── globals.css      # Tailwind + custom styles
├── components/          # React components
│   ├── Navigation.tsx   # Tab navigation
│   ├── StatsCard.tsx    # Stat display cards
│   ├── SpeedupChart.tsx # Bar chart for speedups
│   ├── StatusChart.tsx  # Pie chart for status
│   ├── BenchmarkTable.tsx # Sortable/filterable table
│   ├── GpuCard.tsx      # GPU info display
│   └── LLMInsights.tsx  # AI analysis display
├── lib/                 # Utilities
│   ├── api.ts           # API client functions
│   └── utils.ts         # Helper functions
└── types/               # TypeScript types
    └── index.ts         # Type definitions
```

## Features

- **Overview Tab**: Stats, charts, GPU info, benchmark table
- **LLM Insights Tab**: AI-powered analysis and recommendations
- **Compare Tab**: Side-by-side benchmark comparison (coming soon)
- **Roofline Tab**: Roofline model visualization (coming soon)
- **Profiler Tab**: GPU flame graphs and kernels (coming soon)
- **Memory Tab**: Memory timeline analysis (coming soon)
- **Advanced Tab**: Cost calculator, what-if analysis (coming soon)
- **Multi-GPU Tab**: NVLink topology (coming soon)
- **History Tab**: Performance trends (coming soon)

## API Proxy

The Next.js app proxies all `/api/*` requests to the Python backend at `http://localhost:6970`. Configure this in `next.config.js`.

## Styling

Uses Tailwind CSS with custom design tokens matching the original dashboard aesthetic:

- Dark theme with glass-morphism effects
- Gradient backgrounds with subtle animation
- Accent colors: Cyan (`#00f5d4`), Purple (`#9d4edd`), Pink (`#f72585`)
- JetBrains Mono for code, Inter for UI text

