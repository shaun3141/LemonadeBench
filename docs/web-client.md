# LemonadeBench Web Client

The web client provides a human-friendly interface for playing the Lemonade Stand game, designed for collecting human baseline data for RL research.

## Tech Stack

- **React 18** with TypeScript
- **Vite** for build tooling
- **shadcn/ui** component library (Tailwind + Radix)
- **Lucide** icons

## Architecture Overview

```
web-client/src/
├── App.tsx              # Main app container + game state management
├── api.ts               # Server communication layer
├── types.ts             # TypeScript type definitions
├── components/
│   ├── ActionControls.tsx    # Player decision input panel
│   ├── HumanObservation.tsx  # Human-friendly game state display
│   ├── ModelObservation.tsx  # JSON debug view (model format)
│   ├── MarketInsights.tsx    # Demand forecasting + profit projections
│   ├── WeatherIcon.tsx       # Weather display utilities
│   └── ui/                   # shadcn/ui components
└── lib/
    └── utils.ts              # Tailwind merge utilities
```

---

## State Management

The application uses React's built-in state management (no external library). All game state flows through `App.tsx`.

### Primary State Variables

```typescript
// Game observation from server
const [observation, setObservation] = useState<LemonadeObservation | null>(null);

// UI state
const [isModelView, setIsModelView] = useState(false);     // Toggle JSON vs human view
const [isConnected, setIsConnected] = useState(false);     // Server connection status
const [isLoading, setIsLoading] = useState(false);         // API call in progress
const [error, setError] = useState<string | null>(null);   // Error messages

// Game history (for history tab)
const [history, setHistory] = useState<GameHistory[]>([]);

// Day-end stats modal
const [dayEndStats, setDayEndStats] = useState<GameHistory | null>(null);

// Form state (lifted for MarketInsights access)
const [selectedPrice, setSelectedPrice] = useState(75);    // cents
const [selectedCups, setSelectedCups] = useState(30);
```

### State Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                           App.tsx                               │
│                                                                 │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐  │
│  │ observation │────▶│ Observation  │     │ ActionControls  │  │
│  │   (state)   │     │   Display    │     │    (inputs)     │  │
│  └─────────────┘     └──────────────┘     └────────┬────────┘  │
│         ▲                                          │            │
│         │                                          ▼            │
│  ┌──────┴──────┐                          ┌───────────────┐    │
│  │   Server    │◀────────────────────────│   handleAction │    │
│  │  Response   │       POST /api/step     │   (callback)   │    │
│  └─────────────┘                          └───────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Type Definitions

### LemonadeObservation (from server)

```typescript
interface LemonadeObservation {
  // Day info
  day: number;                              // Current day (1-14)
  weather: 'sunny' | 'hot' | 'cloudy' | 'rainy' | 'stormy';
  temperature: number;                       // Fahrenheit
  weather_forecast: string;                  // Tomorrow's weather

  // Financial state (all values in cents)
  cash: number;                             // Available cash
  daily_revenue: number;                    // Yesterday's sales
  daily_costs: number;                      // Yesterday's costs
  daily_profit: number;                     // Yesterday's profit

  // Sales metrics (from yesterday)
  cups_sold: number;
  cups_wasted: number;
  customers_served: number;
  customers_turned_away: number;

  // Current inventory
  lemons: number;                           // Lemons in stock
  sugar_bags: number;                       // Sugar bags
  cups_available: number;                   // Disposable cups
  ice_bags: number;                         // Ice bags (melt overnight)

  // Expiration tracking
  lemons_expiring_tomorrow: number;         // Lemons about to spoil
  ice_expiring_tomorrow: number;            // Always = ice_bags (melts daily)
  lemons_spoiled: number;                   // Lost to spoilage
  ice_melted: number;                       // Lost to melting

  // Performance metrics
  customer_satisfaction: number;            // 0.0 - 1.0
  reputation: number;                       // 0.0 - 1.0 (affects demand)
  days_remaining: number;
  total_profit: number;                     // Season total
  done?: boolean;                           // Game over flag
  reward?: number;                          // RL reward signal

  // Advanced analytics (optional)
  market_hints?: MarketHints;               // Demand forecasting data
}
```

### LemonadeAction (to server)

```typescript
interface LemonadeAction {
  price_per_cup: number;      // Price in cents (25-200)
  cups_to_make: number;       // How many cups to prepare (0-250)
  buy_lemons: number;         // Lemons to purchase
  buy_sugar: number;          // Sugar bags to purchase
  buy_cups: number;           // Disposable cups to purchase
  buy_ice: number;            // Ice bags to purchase (melt overnight!)
  advertising_spend: number;  // Ad budget in cents
}
```

### MarketHints (demand forecasting)

Uses a two-stage model: **Foot Traffic** (people who stop by) × **Conversion Rate** (% who buy) = **Sales**

```typescript
interface MarketHints {
  // Stage 1: Foot traffic (people who stop by the stand)
  foot_traffic_low: number;   // Conservative estimate (with -10% randomness)
  foot_traffic_high: number;  // Optimistic estimate (with +10% randomness)
  weather_traffic_multiplier: number;  // How weather affects foot traffic (0.1-1.8)

  // Stage 2: Conversion rates (% who buy at each price)
  conversion_curve: Record<number, number>;  // price → conversion rate (0.0-1.0)
  ice_conversion_bonus: number;  // Bonus conversion % on hot days when you have ice

  // Derived demand (foot_traffic × conversion for convenience)
  optimal_price: number;
  price_demand_curve: Record<number, number>;  // price → expected sales
  revenue_curve: Record<number, number>;  // price → expected revenue
  optimal_revenue_price: number;  // price that maximizes revenue

  // Inventory insights
  max_cups_producible: number;
  max_cups_with_ice: number;
  limiting_resource: 'lemons' | 'sugar' | 'cups' | 'ice';
  ingredient_cost_per_cup: number;

  // Strategy hints
  break_even_price: number;
  suggested_production: number;
  has_ice: boolean;
  ice_bonus_active: boolean;

  // Constants
  recipe_info: {
    lemons_per_cup: number;
    sugar_per_cup: number;
    ice_per_cup: number;
    cups_from_one_lemon: number;      // 4
    cups_from_one_sugar_bag: number;  // 10
    cups_from_one_ice_bag: number;    // 5
  };
  supply_costs: {
    lemon: number;      // 25 cents
    sugar_bag: number;  // 100 cents
    cup: number;        // 5 cents
    ice_bag: number;    // 50 cents
  };
}
```

---

## Component Reference

### `App.tsx` - Main Container

**Responsibilities:**
- Server connection management (health check polling)
- Game state storage and lifecycle
- Action submission and history tracking
- Modal management (day-end stats, market intel)

**Key Functions:**
```typescript
handleReset()   // POST /api/reset → initialize new game
handleAction()  // POST /api/step → submit day's decisions
```

---

### `ActionControls.tsx` - Decision Input Panel

The main interactive component where players make daily decisions.

**Props:**
```typescript
interface ActionControlsProps {
  observation: LemonadeObservation;
  onSubmit: (action: LemonadeAction) => void;
  onReset: () => void;
  disabled?: boolean;
  selectedPrice: number;
  selectedCups: number;
  onPriceChange: (price: number) => void;
  onCupsChange: (cups: number) => void;
}
```

**Local State:**
- `buyLemons`, `buySugar`, `buyCups`, `buyIce` - purchase quantities
- `advertising` - ad spend amount

**Calculations:**
```typescript
// Production capacity with pending purchases
const totalLemons = observation.lemons + buyLemons;
const totalSugar = observation.sugar_bags + buySugar;
const totalCups = observation.cups_available + buyCups;
const totalIce = (observation.ice_bags || 0) + buyIce;

// Limiting factor calculation
const maxCupsFromLemons = Math.floor(totalLemons * 4);   // 4 cups/lemon
const maxCupsFromSugar = totalSugar * 10;                // 10 cups/bag
const maxCupsFromCups = totalCups;                       // 1:1
const maxPossibleCups = Math.min(maxCupsFromLemons, maxCupsFromSugar, maxCupsFromCups);

// Cost validation
const totalCost = lemonCost + sugarCost + cupCost + iceCost + advertising;
const canAfford = totalCost <= observation.cash;
```

**UI Sections:**
1. **Inventory & Supplies** - 2x2 grid with buy controls for lemons, ice, sugar, cups
2. **Price/Cups/Advertising** - 3-column slider controls
3. **Action Buttons** - Start Day / Reset

---

### `HumanObservation.tsx` - Human-Friendly Display

Renders game state in a visual, dashboard-style layout.

**Sections:**
1. **Welcome Banner** (Day 1 only) - Goal explanation
2. **Day/Weather Card** - Current day, weather, temperature, forecast
3. **Financials Card** - Cash on hand, total profit, yesterday's profit
4. **Reputation Card** - Star rating + progress bar
5. **Yesterday's Results** (Day 2+) - Revenue, costs, sales metrics
6. **Game Over Banner** - Final score display

**Helper Functions:**
```typescript
formatCents(cents: number): string        // "$1.25"
getReputationStars(reputation: number): string  // "⭐⭐⭐☆☆"
getProfitColor(profit: number): string    // Tailwind color class
```

---

### `ModelObservation.tsx` - JSON Debug View

Renders the raw observation as formatted JSON for debugging/model comparison.

```typescript
// Simply displays observation as pretty-printed JSON
<pre className="font-mono text-sm bg-zinc-900 text-green-400">
  {JSON.stringify(observation, null, 2)}
</pre>
```

---

### `MarketInsights.tsx` - Demand Forecasting

Displays market intelligence data from `observation.market_hints`.

**Sections:**
1. **Market Forecast** - Weather demand multiplier, expected customer range
2. **Price Strategy** - Price-demand curve table, optimal price recommendation
3. **Production Guide** - Max capacity, limiting resource, break-even analysis
4. **Profit Projection** - Expected revenue/costs/profit based on current selections

**Key Calculation:**
```typescript
// Projected profit based on player selections
const actualCupsMade = Math.min(selectedCups, hints.max_cups_producible);
const expectedSales = Math.min(actualCupsMade, expectedDemand);
const expectedRevenue = expectedSales * selectedPrice;
const ingredientCosts = actualCupsMade * hints.ingredient_cost_per_cup;
const projectedProfit = expectedRevenue - ingredientCosts;
```

---

### `WeatherIcon.tsx` - Weather Utilities

Exports weather display helpers:

```typescript
WeatherIcon({ weather, size, className })  // Lucide icon component
getWeatherEmoji(weather: string): string   // 🔥 ☀️ ☁️ 🌧️ ⛈️
getWeatherLabel(weather: string): string   // "Hot", "Sunny", etc.
```

---

## API Layer (`api.ts`)

Simple fetch wrapper for server communication:

```typescript
const API_BASE = '/api';

export async function resetGame(): Promise<GameState>
export async function stepGame(action: LemonadeAction): Promise<GameState>
export async function getHealth(): Promise<{ status: string }>
```

Vite proxies `/api/*` to the backend server (configured in `vite.config.ts`).

---

## Game Mechanics (Frontend Reference)

### Recipe Constants

| Ingredient  | Cost   | Yield      | Expiration |
|-------------|--------|------------|------------|
| Lemon       | $0.25  | 4 cups     | 3 days     |
| Sugar Bag   | $0.50  | 10 cups    | Never      |
| Cup         | $0.05  | 1 serving  | Never      |
| Ice Bag     | $0.25  | 5 cups     | **Overnight** |

### Weather Impact

| Weather | Emoji | Demand Multiplier |
|---------|-------|-------------------|
| Hot     | 🔥    | 1.8x              |
| Sunny   | ☀️    | 1.3x              |
| Cloudy  | ☁️    | 0.9x              |
| Rainy   | 🌧️    | 0.4x              |
| Stormy  | ⛈️    | 0.1x              |

### Ice Bonus

When ice is available and weather is hot/sunny:
- **+20% customer demand** boost
- Ice melts completely overnight (buy fresh daily)

### Pricing Psychology

- **Optimal price**: ~$0.50 maximizes customer volume
- **Above $1.00**: Significant demand drop-off
- **Above $2.00**: Almost no customers

---

## Running the Web Client

```bash
cd web-client

# Install dependencies
npm install

# Start dev server (with API proxy to localhost:8000)
npm run dev

# Build for production
npm run build
```

**Note:** The backend server must be running on port 8000 for the game to work:

```bash
cd lemonade_bench
uv run server --host 0.0.0.0 --port 8000
```
