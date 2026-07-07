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
│   ├── dashboard.ts        # Dashboard API functions
│   ├── datasets.ts         # Dataset API functions
│   ├── clients.ts          # Client API functions
│   └── training.ts         # Training API functions
├── components/
│   ├── layout/             # Header, Sidebar, MainLayout
│   └── ui/                 # Card, StatCard, StatusBadge
├── pages/                  # Route pages
│   ├── Dashboard.tsx       # Dashboard with React Query, auto-refresh, error/loading states
│   ├── Clients.tsx
│   ├── Datasets.tsx
│   └── Training.tsx        # Full training dashboard with live backend integration
├── services/               # Legacy mock services (migrated to api/)
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

### Training API (`src/api/training.ts`)

Eight endpoints:

| Function | Method | Endpoint | Purpose |
|---|---|---|---|
| `fetchTrainingStatus()` | GET | `/api/v1/training/status` | Current training state, metrics, and progress |
| `fetchTrainingConfig()` | GET | `/api/v1/training/config` | Training hyperparameter configuration |
| `startTraining()` | POST | `/api/v1/training/start` | Begin training run |
| `pauseTraining()` | POST | `/api/v1/training/pause` | Pause active training |
| `resumeTraining()` | POST | `/api/v1/training/resume` | Resume paused training |
| `stopTraining()` | POST | `/api/v1/training/stop` | Stop training entirely |
| `saveCheckpoint()` | POST | `/api/v1/training/checkpoint` | Save a training checkpoint |
| `updateTrainingConfig(data)` | PUT | `/api/v1/training/config` | Update training configuration |

All functions return typed responses matching backend Pydantic schemas.

## Training Page Flow

1. **Mount**: `useQuery` fires `fetchTrainingStatus()` via Axios
2. **Loading**: Skeleton cards + spinner for charts
3. **Success**: Full training dashboard with status cards, controls, config panel, progress bars, and round history
4. **Error**: Friendly error UI with icon, message, and "Try Again" button
5. **Empty**: "No training data available" state with Retry

### Configuration Flow

1. Click "Config" button to toggle the configuration panel open
2. `fetchTrainingConfig()` loads current parameters from the backend
3. Edit fields in the form (text, number, select, toggle)
4. Client-side validation runs on save:
   - Learning rate > 0
   - Client count ≥ 1
   - Communication rounds ≥ 1
   - Local epochs ≥ 1
   - Batch size ≥ 1
   - Dataset name required
5. Click "Save Configuration" → `updateTrainingConfig()` mutation
6. Success invalidates both config and status queries
7. "Reset" reverts to the last loaded values
8. "Reload Config" refetches from backend

### Polling Strategy

- Status query uses conditional `refetchInterval`:
  - **2 seconds** when training status is `"running"`
  - **Disabled** (no polling) when idle, paused, completed, or failed
- Interval is evaluated reactively via React Query's callback form: `refetchInterval: (query) => query.state.data?.data?.status === 'running' ? 2000 : false`
- Automatic stop when training completes or pauses (no wasted requests)
- Manual "Refresh" button always available for on-demand updates

### Training Controls

Button enable/disable logic based on current training phase:

| Button | Idle | Running | Paused | Completed | Failed |
|---|---|---|---|---|---|
| Start | ✅ | ❌ | ❌ | ✅ | ✅ |
| Pause | ❌ | ✅ | ❌ | ❌ | ❌ |
| Resume | ❌ | ❌ | ✅ | ❌ | ❌ |
| Stop | ❌ | ✅ | ✅ | ❌ | ❌ |
| Checkpoint | ❌ | ✅ | ❌ | ❌ | ❌ |
| Reload Config | ✅ | ✅ | ✅ | ✅ | ✅ |

- All buttons disable during any pending mutation (prevents double-clicks)
- Mutations invalidate the status query on success to trigger immediate refresh
- Success/error notifications appear as toast in the bottom-right corner (auto-dismiss after 5 seconds)

### Convergence Data

- Accumulated in-memory from polled status responses
- New data point appended when `current_round` changes
- Displayed in the Convergence Analysis chart and Round Execution History table
- Lost on page refresh (repopulates as polling resumes)

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

### Training test coverage:
- Loading skeleton renders
- Error states (server error, backend unavailable)
- Empty data state
- Stat cards with all phases (idle, running, paused, completed)
- Current status section
- Live progress bars and metrics
- Manual refresh button
- Disabled button states for all 5 phases
- Start/Pause/Resume/Stop/Checkpoint mutation calls
- Success notifications
- Error notifications
- Configuration panel open/close
- Configuration load from backend
- Configuration save mutation
- Input validation (learning rate)
- Convergence data accumulation
- Round execution history table
- Empty history state
- Round tracking across status updates
