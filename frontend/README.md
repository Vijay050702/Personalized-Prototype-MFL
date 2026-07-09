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
│   ├── auth.ts             # Auth API + token storage abstraction
│   ├── axios.ts            # Reusable Axios client (base URL, interceptors, error handling, auth)
│   ├── dashboard.ts        # Dashboard API functions
│   ├── datasets.ts         # Dataset API functions
│   ├── clients.ts          # Client API functions
│   ├── training.ts         # Training API functions
│   ├── evaluation.ts       # Evaluation API functions
│   ├── experiments.ts      # Experiments API functions
│   ├── knowledgeTransfer.ts # Knowledge Transfer API functions
│   ├── prototypes.ts       # Prototype API functions
│   ├── settings.ts         # Settings API functions
│   └── similarity.ts       # Similarity API functions
├── components/
│   ├── auth/               # AuthBanner, LoginPage
│   ├── layout/             # Header, Sidebar, MainLayout
│   ├── realtime/           # LiveDashboard page component
│   └── ui/                 # Card, StatCard, StatusBadge
├── context/
│   └── AuthContext.tsx      # Session provider & auth state management
├── hooks/
│   └── useAuth.ts           # useAuth() convenience hook
├── realtime/               # Realtime monitoring layer (types, events, connection, provider, hooks, status)
├── pages/                  # Route pages (10 pages)
├── routes/
│   └── ProtectedRoute.tsx   # Route guard component
├── services/               # Legacy mock services (migrated to api/)
├── test/                   # Test infrastructure
│   ├── setup.ts            # Vitest setup (jest-dom matchers)
│   └── __tests__/          # Component tests (11 test files, 322 tests)
├── types.ts                # Shared TypeScript interfaces
├── App.tsx                 # Router setup with ProtectedRoute wrappers
├── main.tsx                # Entry point with QueryClientProvider + AuthProvider
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

### Total test count: **322 tests across 11 test files**

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

### Knowledge Transfer test coverage (27 tests):
- Loading skeleton renders
- Error states (server error, backend unavailable)
- Error retry button
- Empty data state with refresh
- Statistics cards with backend data and fallback values
- Data table with row rendering and mount call
- Auto-refresh behavior
- Sorting (single click changes order, double click reverses)
- Filtering (status select, clear filters)
- Pagination (many items, page size selector)
- Detail panel (open, section headings present, similarity metrics, close)
- Visualizations (Cross-Modal Transfer Graph, Similarity Heatmap, Transfer Timeline, Transfer Success Distribution, Transfer Loss Curve, Modality Interaction Matrix)
- Last updated timestamp

## Authentication Infrastructure

The frontend includes a complete authentication system that **automatically adapts** to the backend's capabilities.

### Architecture

```
src/
├── api/
│   └── auth.ts              # Auth API + token storage abstraction
├── context/
│   └── AuthContext.tsx       # Session provider + state management
├── hooks/
│   └── useAuth.ts            # useAuth() convenience hook
├── components/
│   └── auth/
│       ├── AuthBanner.tsx    # "Auth Disabled" banner component
│       ├── LoginPage.tsx     # Login page (falls back to disabled message)
│       └── index.ts          # Barrel exports
├── routes/
│   └── ProtectedRoute.tsx    # Route guard component
```

### How It Works

1. **Probe on mount**: `AuthContext` calls `probeAuthEndpoints()` which pings `GET /api/v1/auth/status`
2. **If backend has auth** (200): Enables full login/session flow
3. **If backend has no auth** (404/error): Enters **Disabled Authentication Mode**

### Disabled Authentication Mode

When the backend does not expose authentication endpoints:
- All routes remain accessible
- An `AuthBanner` appears at the top of protected pages: "Authentication is not enabled by the backend"
- The `/login` page renders an informative message instead of a login form
- The `useAuth()` hook returns `{ isAuthEnabled: false, mode: 'disabled' }`
- No session management, token storage, or login/logout operations occur
- Existing UI and functionality are unchanged

### Protected Routes

Routes wrapped with `<ProtectedRoute>`:

| Route | Page | Behavior (no auth) |
|---|---|---|
| `/` | Dashboard | Renders content + AuthBanner |
| `/clients` | Clients | Renders content + AuthBanner |
| `/datasets` | Datasets | Renders content + AuthBanner |
| `/training` | Training | Renders content + AuthBanner |
| `/prototypes` | Prototypes | Renders content + AuthBanner |
| `/knowledge-transfer` | Knowledge Transfer | Renders content + AuthBanner |
| `/similarity` | Similarity | Renders content + AuthBanner |
| `/evaluation` | Evaluation | Renders content + AuthBanner |
| `/experiments` | Experiments | Renders content + AuthBanner |
| `/settings` | Settings | Renders content + AuthBanner |
| `/live` | Live Monitor | Renders content + AuthBanner (realtime dashboard) |
| `/login` | Login | Displays auth-disabled message |

### Session Flow (when auth is enabled)

```
App Start
   │
   ▼
Probe GET /api/v1/auth/status
   │
   ├── 200 ──► Auth Enabled ──► Check stored token
   │                              │
   │                         ┌────┴────┐
   │                         ▼         ▼
   │                    Has Token    No Token
   │                         │          │
   │                    GET /me    ┌────┘
   │                         │    │
   │                    ┌────┴┐   │
   │                    ▼    ▼    │
   │                 Valid  Invalid
   │                  │       │
   │               ┌──┘    Logout
   │               ▼
   │           Authenticated
   │
   └── 404 ──► Auth Disabled ──► Allow access + show banner
```

### Token Storage

Abstract storage layer with three implementations:

| Storage | Used When | Persistence |
|---|---|---|
| `localStorage` | Default (backend detected) | Survives tab close |
| `sessionStorage` | Explicit config | Per-tab session |
| `memory` | Fallback (localStorage unavailable) | Page reload clears |

The storage layer provides: `getAccessToken()`, `setAccessToken()`, `getRefreshToken()`, `setRefreshToken()`, `getStoredUser()`, `setStoredUser()`, `clear()`.

### Axios Integration

- **Request interceptor**: Injects `Authorization: Bearer <token>` from token storage (with backward compatibility for legacy `auth_token` key)
- **Response interceptor**: Handles:
  - `401` → Dispatches `auth:session-expired` custom event → triggers session clear
  - `403` → Dispatches `auth:forbidden` custom event
  - Timeout / network / 404 / 500 errors (unchanged from existing behavior)

### Token Refresh

When auth is enabled and a login succeeds with `expires_in`, a timer schedules automatic token refresh 60 seconds before expiry using the refresh token.

### Usage

```tsx
import { useAuth } from '../hooks/useAuth';

function MyComponent() {
  const { isAuthEnabled, isAuthenticated, user, login, logout } = useAuth();

  if (!isAuthEnabled) {
    return <p>Auth not configured by backend</p>;
  }

  if (isAuthenticated) {
    return <p>Welcome, {user?.username}</p>;
  }

  return <button onClick={() => login({ username, password })}>Log In</button>;
}
```

### Testing

Auth test coverage (28 tests):

- TokenStorage (localStorage, sessionStorage, memory, defaults)
- AuthContext (probe, disabled mode, session restore, invalid token, network failure, login flow)
- ProtectedRoute (loading, disabled mode, authenticated)
- AuthBanner renders correct message
- LoginPage renders correct placeholder
- useAuth (throws outside provider, returns context inside provider)
- Axios interceptor event dispatch
- Auth API functions (probeAuthEndpoints, clearAuthData)

## Knowledge Transfer Page

The Knowledge Transfer page (`/knowledge-transfer`) displays cross-modal knowledge transfer operations across clients and modalities.

### API Layer (`src/api/knowledgeTransfer.ts`)

Six functions:

| Function | Method | Endpoint | Purpose |
|---|---|---|---|
| `fetchKnowledgeTransfers()` | GET | `/api/v1/knowledge-transfer` | List all knowledge transfer records |
| `fetchKnowledgeTransferDetail(id)` | GET | `/api/v1/knowledge-transfer/{id}` | Single transfer details |
| `fetchKnowledgeTransferStatistics()` | GET | `/api/v1/knowledge-transfer/statistics` | Aggregate statistics |
| `startKnowledgeTransfer(data)` | POST | `/api/v1/knowledge-transfer/start` | Initiate a new transfer |
| `stopKnowledgeTransfer(id)` | POST | `/api/v1/knowledge-transfer/{id}/stop` | Stop a running transfer |
| `fetchKnowledgeTransferHistory()` | GET | `/api/v1/knowledge-transfer/history` | Transfer history timeline |

All functions return typed responses matching the `KnowledgeTransfer*Response` interfaces in `types.ts`.

### Page Features

1. **Header**: Title, description, last-updated timestamp, manual refresh button
2. **Search**: Text input filtering by transfer ID, client, modality, strategy
3. **Filters**: 7 dropdowns — Status, Strategy, Source Client, Target Client, Source Modality, Target Modality, Communication Round range
4. **Statistics Cards**: 8 cards — Total Transfers, Successful, Failed, Avg Similarity, Avg Confidence, Avg Loss, Avg Exec Time, Comm Efficiency
5. **Sortable Data Table**: 10 sortable columns with ascending/descending toggle
6. **Pagination**: Configurable page size (5/10/20/50) with prev/next navigation
7. **Visualizations** (6 charts):
   - **Cross-Modal Transfer Graph**: Bar chart — transfer count by modality pair
   - **Similarity Heatmap**: Bar chart — average similarity by modality pair
   - **Transfer Timeline**: Line chart — similarity and transfer count per round
   - **Transfer Success Distribution**: Pie chart — transfers by status
   - **Transfer Loss Curve**: Area chart — transfer loss across transfers
   - **Modality Interaction Matrix**: Table — average similarity by source-target modality with color intensity
8. **Detail Panel**: Slide-in overlay with Transfer Metadata, Alignment Information (cross-modal mapping, alignment method), Similarity Metrics (score, confidence, loss, execution time), Associated Prototypes, Knowledge Transfer History (communication round, created time)
9. **Auto-Refresh**: Polls every 5 seconds; pauses when browser tab is hidden (Page Visibility API)

### Data Flow

1. **Mount**: Two `useQuery` hooks fire `fetchKnowledgeTransfers()` and `fetchKnowledgeTransferStatistics()`
2. **Loading**: Skeleton cards pulse for statistics area
3. **Success**: Full page with filters, table, stats, charts
4. **Error**: Friendly UI with icon, error message, and "Try Again" button
5. **Empty**: "No knowledge transfers found" message with "Refresh" button

### Types (`src/types.ts`)

| Type | Description |
|---|---|
| `KnowledgeTransferResponse` | Single transfer record (id, clients, modalities, strategy, similarity, confidence, loss, status, etc.) |
| `KnowledgeTransferListResponse` | Paginated list wrapper |
| `KnowledgeTransferStatistics` | Aggregate stats (totals, averages, efficiency) |
| `KnowledgeTransferStatisticsResponse` | Statistics API response wrapper |
| `KnowledgeTransferStartRequest` | Request body for starting a transfer |
| `KnowledgeTransferHistoryResponse` | History timeline wrapper |

## Evaluation Page

The Evaluation page (`/evaluation`) displays live model evaluation metrics from the backend.

### API Layer (`src/api/evaluation.ts`)

| Function | Method | Endpoint | Purpose |
|---|---|---|---|
| `fetchEvaluation()` | GET | `/api/v1/evaluation` | Core evaluation metrics (accuracy, precision, recall, f1, auc_roc) |
| `fetchExperiments()` | GET | `/api/v1/experiments` | Experiment runs for baseline comparison and Experiments page |

### Data Flow

1. **Mount**: Two `useQuery` hooks fire `fetchEvaluation()` and `fetchExperiments()` in parallel
2. **Loading**: Skeleton cards + spinner placeholders for charts
3. **Success**: Full evaluation dashboard with metrics, charts, baseline comparison, and experiment table
4. **Error**: Friendly UI with icon, message, and "Try Again" button
5. **Empty**: "No evaluation data available" state with Retry

### Displayed Metrics (from `/api/v1/evaluation`)

- **Accuracy** — Overall classification accuracy
- **Precision** — Weighted precision score
- **Recall** — Weighted recall score
- **F1 Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Area under the ROC curve
- **Comm Round** — Current communication round
- **Samples** — Number of samples evaluated
- **Client** — Client ID (global/personalized)

### Visualizations

1. **Precision / Recall / F1 Bar Chart** — Side-by-side bar comparison of three key metrics
2. **Performance Radar Chart** — Five-metric radar (accuracy, precision, recall, f1, auc_roc)
3. **Baseline Comparison Chart** — Horizontal bar chart comparing best accuracy across algorithms (FedAvg, FedProx, SCAFFOLD, pFedProto, etc.) from experiment runs
4. **Best Model Highlight** — Green callout card identifying the experiment with highest accuracy

### Baseline Comparison

The Baseline Comparison section uses experiment data from `GET /api/v1/experiments`:
- Groups experiments by algorithm
- Takes the best accuracy per algorithm
- Renders a horizontal bar chart sorted by algorithm type
- Highlights the overall best model in a callout card

### Experiment Table

| Feature | Description |
|---|---|
| **Search** | Text search across name, ID, and algorithm |
| **Status Filter** | Dropdown to filter by running/completed/pending/failed |
| **Algorithm Filter** | Dropdown to filter by unique algorithms |
| **Clear Filters** | Resets all search/filter state |
| **Sorting** | Click column headers to sort ascending/descending |
| **Detail Panel** | Click the eye icon to open a slide-in panel with experiment metadata, performance, and timeline |
| **Empty State** | Shows "No experiments match your filters" when filtered results are empty |

### Polling Strategy

- **Evaluation query**: Auto-refreshes every 10 seconds
- **Experiments query**: Auto-refreshes every 30 seconds
- Manual "Refresh" button always available for on-demand updates

### React Query Integration

- `queryKey: ['evaluation']` for core metrics
- `queryKey: ['experiments']` for experiment runs
- `staleTime` configured per query to balance freshness vs. request volume
- Both queries use `retry: 2` for transient error resilience

### Evaluation test coverage (30+ tests):
- Loading skeleton renders
- Error states (server error, backend unavailable, 404, 422)
- Error retry button
- Empty data state with Retry
- Stat cards with backend evaluation metrics (accuracy, precision, recall, f1, auc_roc)
- Evaluation metadata section (client ID, comm round, samples)
- Charts render (Precision/Recall/F1 bar chart, Performance radar chart)
- Baseline comparison with best model highlight
- Experiment table renders all rows
- Experiment status badges
- Search filtering
- Status filter
- Algorithm filter
- Clear filters
- No results state
- Detail panel open/close
- Sorting by column click
- Manual refresh button
- Last updated timestamp
- Multiple query error handling

## Realtime Monitoring Layer

A unified real-time monitoring layer (`src/realtime/`) that automatically detects the highest-capability transport supported by the backend and provides typed event tracking, connection status indicators, and a live monitoring dashboard.

### Architecture

```
src/realtime/
├── types.ts              # All type definitions (TransportType, ConnectionStatus, RealtimeEvent, etc.)
├── events.ts             # Event creation, filtering, and severity utilities
├── connection.ts         # Transport detection probes (WebSocket → SSE → polling)
├── provider.tsx          # RealtimeContext + RealtimeProvider (wraps app, drives all state)
├── hooks.ts              # React hooks: useRealtime, useConnectionStatus, useEventHistory, useRealtimeDashboard
└── status.tsx            # UI components: ConnectionStatusIndicator, TransportLabel
```

### Transport Detection Strategy

On mount, the `RealtimeProvider` probes transports in priority order:

1. **WebSocket** — Attempts `ws://localhost:8000/ws/health` with 3s timeout
2. **SSE** — Attempts `GET /api/v1/events/stream` with `Accept: text/event-stream`
3. **Polling** — Attempts `GET /api/v1/dashboard` via Axios (guaranteed fallback while backend is reachable)
4. **None** — All probes failed; backend unreachable

Detection runs once on mount (guarded by a `probeDone` ref). The result determines the connection status displayed throughout the app.

### Provider Integration

```tsx
<QueryClientProvider client={queryClient}>
  <AuthProvider>
    <RealtimeProvider pollingInterval={5000}>
      <App />
    </RealtimeProvider>
  </AuthProvider>
</QueryClientProvider>
```

`RealtimeProvider` wraps `AuthProvider` (to sit above routes) but is nested inside `QueryClientProvider` (it uses `useQueryClient` to call `invalidateQueries`).

### Context Value

| Property | Type | Description |
|---|---|---|
| `transport` | `'websocket' \| 'sse' \| 'polling' \| 'none'` | Detected transport |
| `connectionStatus` | `'connecting' \| 'connected' \| 'polling' \| 'offline' \| 'disconnected'` | Current status |
| `events` | `RealtimeEvent[]` | Event history (auto-pruned at 500) |
| `isLive` | `boolean` | `true` when connected or polling |
| `transportError` | `string \| null` | Last probe error |
| `currentTransportLabel` | `string` | Human-readable transport name |
| `clearHistory()` | `() => void` | Clears all events |
| `removeEvent(id)` | `(id: string) => void` | Removes a single event |
| `refetchAll()` | `() => void` | Invalidates all known React Query keys |

### Hooks

| Hook | Returns | Purpose |
|---|---|---|
| `useRealtime()` | Full `RealtimeContextValue` | Access all realtime state and actions |
| `useConnectionStatus()` | `{ connectionStatus, transport, currentTransportLabel, isLive }` | Status display helpers |
| `useEventHistory(opts?)` | `RealtimeEvent[]` | Filtered event list by category, severity, search, limit |
| `useRealtimeDashboard(refetchInterval?)` | `{ data, isLoading, error, refetch }` | Wraps `fetchDashboard` with configurable polling interval (default 5s) |

### Route

- `/live` — `LiveDashboard` component rendered via `<ProtectedRoute>`

### LiveDashboard Page

A full-page live monitor with:

1. **Header** — Title, connection status indicator, transport label, event count, last-updated timestamp, Refresh and Clear buttons
2. **Stat Cards** (8) — Current Round, Running Experiments, Active Clients, Communication Rate, Prototype Updates, KT Events, Global Accuracy, Training Loss
3. **Server Status Panel** — Transport, Training Status, Uptime, Communication Round, Last Poll Timestamp
4. **Event History Panel** — Filterable (category/severity/search), auto-scrolling, auto-pruned at 500 entries

### UI Components

| Component | Props | Description |
|---|---|---|
| `ConnectionStatusIndicator` | `showLabel?: boolean`, `showTransport?: boolean` | Colored ping dot + status label |
| `TransportLabel` | — | Transport type label |

Status color mapping:

| Status | Color |
|---|---|
| connected | Green (emerald) |
| polling | Blue (sky) |
| connecting | Yellow (amber) |
| disconnected | Gray (slate) |
| offline | Red (rose) |

### Realtime test coverage (36 tests):
- Event utilities (createEvent, severityFromStatus, filterEvents, getCategoryLabel, TRANSPORT_LABELS)
- detectTransport mock integration
- createConnectionManager (initial state, connect, disconnect)
- RealtimeProvider (initial context, transition to polling, transition to offline, clearHistory, removeEvent, events populated after probe, refetchAll)
- useConnectionStatus returns correct values
- useEventHistory returns filtered
- useRealtimeDashboard (loading, data)
- ConnectionStatusIndicator renders with/without label
- TransportLabel renders
- LiveDashboard (loading, heading, stat values, buttons, sections, filter toggle, transport display)
- useRealtime throws outside provider
- MAX_HISTORY_SIZE constant

## Experiments Page

The Experiments page (`/experiments`) provides a comprehensive view of all experiment runs in the system.

### API Layer (`src/api/experiments.ts`)

| Function | Method | Endpoint | Purpose |
|---|---|---|---|
| `fetchExperiments()` | GET | `/api/v1/experiments` | All experiment runs with status, accuracy, rounds, etc. |

### Data Flow

1. **Mount**: `useQuery` fires `fetchExperiments()` via Axios
2. **Loading**: 8 skeleton cards + spinner placeholders for charts
3. **Success**: Full experiments dashboard with stats, charts, timeline, and table
4. **Error**: Friendly UI with icon, message, and "Try Again" button  
5. **Empty**: "No experiments available" state with Retry

### Statistics Cards (8 cards)

- **Total** — Total number of experiments
- **Running** — Currently active experiments
- **Completed** — Successfully finished experiments
- **Failed** — Failed experiments
- **Avg Accuracy** — Average best accuracy across all experiments
- **Avg Duration** — Average runtime duration
- **Best Experiment** — Name of the highest-accuracy experiment
- **Most Used Algo** — Most frequently used algorithm

### Visualizations (4 charts)

1. **Algorithm Comparison** — Dual-axis bar chart showing experiment count and average accuracy per algorithm
2. **Duration Distribution** — Bar chart showing runtime buckets (<1h, 1-6h, 6-24h, 1-3d, >3d)
3. **Status Distribution** — Donut/pie chart showing experiment status breakdown (completed, running, pending, failed)
4. **Accuracy Trend** — Line chart showing best accuracy across experiments chronologically

### Timeline

- Chronological view of all experiments with colored status indicators
- Shows experiment name, status badge, date range, and best accuracy
- Running experiments show an animated pulse indicator and "Present" end date

### Experiment Table

| Feature | Description |
|---|---|
| **Search** | Text search across name, ID, and algorithm |
| **Status Filter** | Dropdown to filter by running/completed/pending/failed |
| **Algorithm Filter** | Dropdown to filter by unique algorithms |
| **Page Size** | Configurable rows per page (5/10/20/50) |
| **Pagination** | Page navigation with numbered buttons and ellipsis |
| **Sorting** | Click column headers to sort ascending/descending (name, algorithm, status, clients, rounds, accuracy, started) |
| **Clear Filters** | Resets all search/filter state |
| **Detail Panel** | Click the eye icon to open a slide-in panel with metadata, performance, timeline, and log viewer |

### Detail Panel

The slide-in detail panel contains:

- **Metadata** — ID, name, status, algorithm, clients
- **Performance** — Best accuracy, round progress
- **Timeline** — Started and completed timestamps
- **Log Viewer** — Tabbed log interface with 5 tabs:
  - **Training Logs** — Round-level training progress messages
  - **System Logs** — Experiment lifecycle events (initialize, complete, fail)
  - **Aggregation Logs** — Round-level aggregation messages
  - **Knowledge Transfer Logs** — Placeholder for cross-modal transfer events
  - **Personalization Logs** — Placeholder for personalization events
  - Log entries are generated from experiment metadata (status transitions, round changes)

### Polling Strategy

- Auto-refreshes every **10 seconds** when any experiment has `status === 'running'`
- Polling stops (no interval) when all experiments are idle/completed/failed/pending
- Uses React Query's callback form: `refetchInterval: (query) => hasRunning ? 10000 : false`
- Manual "Refresh" button always available for on-demand updates

### Experiments test coverage:
- Loading skeleton renders (8 skeleton cards + chart spinners)
- Error states (server error, backend unavailable, 404, 422, 500, timeout)
- Error retry functionality
- Empty data state with retry
- Statistics cards (total, running, completed, failed, avg accuracy, best experiment, most used algorithm)
- Charts render (algorithm comparison, duration distribution, status distribution pie, accuracy trend)
- Timeline section with experiment events
- Table renders all experiment rows
- Experiment status badges
- Search filtering
- Status filter
- Algorithm filter
- Page size selector and pagination
- Clear filters
- No results state
- Detail panel open/close
- Detail logs tab switching
- Sorting by column click
- Manual refresh button
- Last updated timestamp
- Auto-refresh indicator for running experiments
