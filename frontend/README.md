# PP-MFL Frontend

Frontend dashboard for the Personalized Prototype-Based Multimodal Federated Learning system.

## Tech Stack

- **React 19** with TypeScript
- **Vite 6** build tool
- **Tailwind CSS v4** styling
- **@tanstack/react-query** data fetching & caching
- **Axios** HTTP client
- **Recharts** charts
- **Lucide React** icons
- **React Router v7** routing

## Project Structure

```
src/
├── api/                    # API layer
│   ├── axios.ts            # Reusable Axios client (base URL, interceptors, error handling)
│   └── dashboard.ts        # Dashboard API functions
├── components/
│   ├── layout/             # Header, Sidebar, MainLayout
│   └── ui/                 # Card, StatCard, StatusBadge
├── pages/                  # Route pages
│   ├── Dashboard.tsx       # Dashboard with React Query, auto-refresh, error/loading states
│   ├── Clients.tsx
│   ├── Datasets.tsx
│   └── Training.tsx
├── services/               # Legacy mock services (being migrated to api/)
├── test/                   # Test infrastructure
│   ├── setup.ts            # Vitest setup (jest-dom matchers)
│   └── __tests__/          # Component tests
├── types.ts                # Shared TypeScript interfaces
├── App.tsx                 # Router setup
├── main.tsx                # Entry point with QueryClientProvider
└── index.css               # Global styles & Tailwind theme
```

## API Layer

The `src/api/` directory replaces the legacy `src/services/` mock layer.

### Axios Client (`src/api/axios.ts`)

- Base URL from `VITE_API_URL` env var (default `http://localhost:8000`)
- 15-second request timeout
- JSON content type headers
- Request interceptor for future JWT auth (reads `auth_token` from localStorage)
- Response interceptor handling:
  - Timeout errors → `"Request timed out. Please try again."`
  - Network errors → `"Backend is unavailable. Please check your connection."`
  - 404 → `"Resource not found."`
  - 500+ → `"Server error. Please try again later."`

### Dashboard API (`src/api/dashboard.ts`)

```typescript
import { fetchDashboard } from '../api/dashboard';
const summary = await fetchDashboard();
// summary.data.active_clients, summary.data.global_accuracy, etc.
```

- Endpoint: `GET /api/v1/dashboard`
- Returns typed `DashboardSummary` matching backend Pydantic schema

## Dashboard Page Flow

1. **Mount**: `useQuery` fires `fetchDashboard()` via Axios
2. **Loading**: Skeleton cards (4 animated pulse placeholders) + spinner on charts
3. **Success**: 4 stat cards (Active Clients, Current Round, Global Accuracy, Training Loss) + System Status panel + Round Progress
4. **Error**: Friendly error UI with icon, message, and "Try Again" button
5. **Empty**: "No dashboard data available" state with Retry
6. **Auto-refresh**: Every 5 seconds (`refetchInterval: 5000`)
7. **Manual refresh**: Header button with spinning icon during refetch

## Configuration

Create a `.env` file in the frontend root:

```
VITE_API_URL=http://localhost:8000
```

No hardcoded `localhost` URLs exist in the source code.

## Running Locally

```bash
npm install
npm run dev          # Dev server on port 3000
npm run build        # Production build
npm run preview      # Preview production build
```

## Testing

```bash
npm test             # Run all tests (vitest run)
npm run test:watch   # Watch mode
```

Tests use **Vitest** + **@testing-library/react** with mocked Axios.

### Dashboard test coverage:
- Loading skeleton renders
- Stat cards with real API data
- System status section
- Round progress section
- Error states (server error, backend unavailable)
- Empty data state
- API call on mount
- Retry on error (Try Again button)
- Manual refresh
- Last updated timestamp
