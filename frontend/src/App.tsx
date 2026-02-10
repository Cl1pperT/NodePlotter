import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import type { LatLngLiteral } from 'leaflet';
import L from 'leaflet';
import { ImageOverlay, MapContainer, Marker, Rectangle, TileLayer, useMap, useMapEvents } from 'react-leaflet';
const MARKER_COLORS = ['#6D28D9', '#2563EB', '#16A34A', '#EAB308', '#F97316', '#DC2626'];

const markerIconCache = new Map<string, L.Icon>();

const getColoredMarkerIcon = (color: string) => {
  const cached = markerIconCache.get(color);
  if (cached) {
    return cached;
  }

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="30" height="45" viewBox="0 0 30 45">
      <path d="M15 0C6.7 0 0 6.7 0 15c0 12 15 30 15 30s15-18 15-30C30 6.7 23.3 0 15 0z" fill="${color}" />
      <circle cx="15" cy="15" r="5.5" fill="#ffffff" fill-opacity="0.85" />
    </svg>
  `.trim();
  const icon = L.icon({
    iconUrl: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`,
    iconSize: [30, 45],
    iconAnchor: [15, 44],
    popupAnchor: [0, -36],
  });
  markerIconCache.set(color, icon);
  return icon;
};

const DEFAULT_CENTER: LatLngLiteral = { lat: 20, lng: 0 };
// Approx. 25 miles across on a typical laptop viewport.
const DEFAULT_ZOOM = 11;
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const WARN_CELL_COUNT = 1_000_000;
const MAX_CELL_COUNT = 4_000_000;
const MAX_GRID_SIDE = 2000;
const FAST_MS_PER_CELL = 0.00311;
const ACCURATE_MS_PER_CELL = 0.183;

const formatDuration = (seconds: number) => {
  if (!Number.isFinite(seconds)) {
    return '—';
  }
  if (seconds < 1) {
    return `${Math.round(seconds * 1000)} ms`;
  }
  if (seconds < 60) {
    return `${seconds.toFixed(1)} s`;
  }
  if (seconds < 3600) {
    return `${(seconds / 60).toFixed(1)} min`;
  }
  return `${(seconds / 3600).toFixed(1)} hr`;
};

type ObserverState = {
  lat: number;
  lng: number;
};

type ConsideredBounds = {
  minLat: number;
  minLon: number;
  maxLat: number;
  maxLon: number;
};

type ParamsState = {
  observerHeightMeters: string;
  maxRadiusKm: string;
  resolutionMeters: string;
};

type FieldErrors = Partial<Record<keyof ParamsState | 'observer' | 'observers' | 'guardrail', string>>;

type OverlayPayload = {
  pngBase64: string;
  boundsLatLon: [number, number, number, number];
};

type ComputeMode = 'accurate' | 'fast';

type HistoryItem = {
  cacheKey: string;
  createdAt?: string | null;
  demVersion?: string | null;
  request?: {
    observer?: {
      lat: number;
      lon: number;
    };
    observers?: {
      lat: number;
      lon: number;
    }[];
    observerHeightM?: number;
    maxRadiusKm?: number;
    resolutionM?: number;
    mode?: ComputeMode;
    consideredBounds?: ConsideredBounds;
    curvatureEnabled?: boolean;
  } | null;
  boundsLatLon?: [number, number, number, number] | null;
};

type ScenarioRequest = {
  mapType: 'single' | 'complex';
  mode: ComputeMode;
  observer?: {
    lat: number;
    lon: number;
  } | null;
  observers?: {
    lat: number;
    lon: number;
  }[] | null;
  observerHeightM: number;
  maxRadiusKm: number;
  resolutionM: number;
  consideredBounds?: ConsideredBounds | null;
  cacheKey?: string | null;
  curvatureEnabled?: boolean;
};

type ScenarioItem = {
  id: string;
  name: string;
  createdAt: string;
  request: ScenarioRequest;
};

type Preset = {
  id: string;
  label: string;
  observerHeightMeters: string;
  maxRadiusKm: string;
  resolutionMeters: string;
};

const PRESETS: Preset[] = [
  {
    id: 'fast',
    label: 'Default',
    observerHeightMeters: '1.7',
    maxRadiusKm: '10',
    resolutionMeters: '90',
  },
  {
    id: 'medium',
    label: 'Medium',
    observerHeightMeters: '1.7',
    maxRadiusKm: '20',
    resolutionMeters: '60',
  },
];

function MapClickHandler({ onSelect }: { onSelect: (coords: ObserverState) => void }) {
  useMapEvents({
    click(event) {
      onSelect({ lat: event.latlng.lat, lng: event.latlng.lng });
    },
  });

  return null;
}

function MapViewController({ center }: { center: LatLngLiteral }) {
  const map = useMap();

  useEffect(() => {
    map.setView(center);
  }, [center, map]);

  return null;
}

export default function App() {
  const [observer, setObserver] = useState<ObserverState | null>(null);
  const [multiObservers, setMultiObservers] = useState<ObserverState[]>([]);
  const [isMultiMode, setIsMultiMode] = useState(false);
  const [consideredBounds, setConsideredBounds] = useState<ConsideredBounds | null>(null);
  const [areaDraft, setAreaDraft] = useState<ObserverState | null>(null);
  const [mapTool, setMapTool] = useState<'observer' | 'area'>('observer');
  const [isMapFullscreen, setIsMapFullscreen] = useState(false);
  const [mapCenter, setMapCenter] = useState<LatLngLiteral>(DEFAULT_CENTER);
  const [status, setStatus] = useState<string | null>(null);
  const [params, setParams] = useState<ParamsState>({
    observerHeightMeters: '1.7',
    maxRadiusKm: '25',
    resolutionMeters: '30',
  });
  const [errors, setErrors] = useState<FieldErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [overlay, setOverlay] = useState<OverlayPayload | null>(null);
  const [lastCacheKey, setLastCacheKey] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ completed: number; total: number } | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [isHistoryCollapsed, setIsHistoryCollapsed] = useState(true);
  const [computeMode, setComputeMode] = useState<ComputeMode>('fast');
  const [curvatureEnabled, setCurvatureEnabled] = useState(true);
  const progressPollRef = useRef<number | null>(null);
  const requestAbortRef = useRef<AbortController | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchStatus, setSearchStatus] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [scenarioName, setScenarioName] = useState('');
  const [scenarioStatus, setScenarioStatus] = useState<string | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioItem[]>([]);
  const [isScenarioSaving, setIsScenarioSaving] = useState(false);
  const [isScenarioLoading, setIsScenarioLoading] = useState(false);
  const [isScenarioCollapsed, setIsScenarioCollapsed] = useState(true);

  const matchedPreset = useMemo(() => {
    return (
      PRESETS.find(
        (preset) =>
          preset.observerHeightMeters === params.observerHeightMeters &&
          preset.maxRadiusKm === params.maxRadiusKm &&
          preset.resolutionMeters === params.resolutionMeters
      ) ?? null
    );
  }, [params]);

  const observerForApi = useMemo(
    () =>
      observer
        ? {
            lat: observer.lat,
            lon: observer.lng,
          }
        : null,
    [observer]
  );

  const observersForApi = useMemo(
    () =>
      multiObservers.map((entry) => ({
        lat: entry.lat,
        lon: entry.lng,
      })),
    [multiObservers]
  );

  const payloadPreview = useMemo(() => {
    if (isMultiMode) {
      if (observersForApi.length === 0) {
        return null;
      }
      const payload: Record<string, unknown> = {
        observers: observersForApi,
        observerHeightM: Number(params.observerHeightMeters),
        maxRadiusKm: Number(params.maxRadiusKm),
        resolutionM: Number(params.resolutionMeters),
        curvatureEnabled,
      };
      if (consideredBounds) {
        payload.consideredBounds = consideredBounds;
      }
      return payload;
    }

    if (!observerForApi) {
      return null;
    }

    const payload: Record<string, unknown> = {
      observer: observerForApi,
      observerHeightM: Number(params.observerHeightMeters),
      maxRadiusKm: Number(params.maxRadiusKm),
      resolutionM: Number(params.resolutionMeters),
      curvatureEnabled,
    };
    if (consideredBounds) {
      payload.consideredBounds = consideredBounds;
    }
    return payload;
  }, [consideredBounds, curvatureEnabled, isMultiMode, observerForApi, observersForApi, params]);

  const estimate = useMemo(() => {
    const radiusKm = Number(params.maxRadiusKm);
    const resolutionM = Number(params.resolutionMeters);
    if (!Number.isFinite(radiusKm) || radiusKm <= 0) {
      return null;
    }
    if (!Number.isFinite(resolutionM) || resolutionM <= 0) {
      return null;
    }
    const radiusM = radiusKm * 1000;

    if (consideredBounds) {
      let minX = Number.POSITIVE_INFINITY;
      let minY = Number.POSITIVE_INFINITY;
      let maxX = Number.NEGATIVE_INFINITY;
      let maxY = Number.NEGATIVE_INFINITY;

      const corners = [
        { lat: consideredBounds.minLat, lng: consideredBounds.minLon },
        { lat: consideredBounds.minLat, lng: consideredBounds.maxLon },
        { lat: consideredBounds.maxLat, lng: consideredBounds.minLon },
        { lat: consideredBounds.maxLat, lng: consideredBounds.maxLon },
      ];
      corners.forEach((corner) => {
        const projected = L.CRS.EPSG3857.project(L.latLng(corner.lat, corner.lng));
        minX = Math.min(minX, projected.x);
        minY = Math.min(minY, projected.y);
        maxX = Math.max(maxX, projected.x);
        maxY = Math.max(maxY, projected.y);
      });

      const points = isMultiMode ? multiObservers : observer ? [observer] : [];
      points.forEach((point) => {
        const projected = L.CRS.EPSG3857.project(L.latLng(point.lat, point.lng));
        minX = Math.min(minX, projected.x);
        minY = Math.min(minY, projected.y);
        maxX = Math.max(maxX, projected.x);
        maxY = Math.max(maxY, projected.y);
      });

      if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
        return null;
      }

      const widthM = maxX - minX;
      const heightM = maxY - minY;
      const gridWidth = Math.ceil(widthM / resolutionM) + 1;
      const gridHeight = Math.ceil(heightM / resolutionM) + 1;
      const gridSide = Math.max(gridWidth, gridHeight);
      const cellCount = gridWidth * gridHeight;
      return { gridWidth, gridHeight, gridSide, cellCount };
    }

    if (isMultiMode && multiObservers.length > 0) {
      let minX = Number.POSITIVE_INFINITY;
      let minY = Number.POSITIVE_INFINITY;
      let maxX = Number.NEGATIVE_INFINITY;
      let maxY = Number.NEGATIVE_INFINITY;

      multiObservers.forEach((point) => {
        const projected = L.CRS.EPSG3857.project(L.latLng(point.lat, point.lng));
        minX = Math.min(minX, projected.x - radiusM);
        minY = Math.min(minY, projected.y - radiusM);
        maxX = Math.max(maxX, projected.x + radiusM);
        maxY = Math.max(maxY, projected.y + radiusM);
      });

      const widthM = maxX - minX;
      const heightM = maxY - minY;
      const gridWidth = Math.ceil(widthM / resolutionM) + 1;
      const gridHeight = Math.ceil(heightM / resolutionM) + 1;
      const gridSide = Math.max(gridWidth, gridHeight);
      const cellCount = gridWidth * gridHeight;
      return { gridWidth, gridHeight, gridSide, cellCount };
    }

    const gridSide = Math.ceil((2 * radiusM) / resolutionM) + 1;
    const cellCount = gridSide * gridSide;
    return { gridWidth: gridSide, gridHeight: gridSide, gridSide, cellCount };
  }, [consideredBounds, isMultiMode, multiObservers, observer, params.maxRadiusKm, params.resolutionMeters]);

  const guardrail = useMemo(() => {
    if (!estimate) {
      return { blocked: false, warnings: [] as string[] };
    }
    const warnings: string[] = [];
    if (estimate.cellCount > WARN_CELL_COUNT) {
      warnings.push(
        `Large request: ${estimate.gridSide}x${estimate.gridSide} (~${estimate.cellCount.toLocaleString()} cells).`
      );
    }
    const blocked = estimate.cellCount > MAX_CELL_COUNT || estimate.gridSide > MAX_GRID_SIDE;
    return { blocked, warnings };
  }, [estimate]);

  const estimatedSeconds = useMemo(() => {
    if (!estimate) {
      return null;
    }
    const perCellMs = computeMode === 'fast' ? FAST_MS_PER_CELL : ACCURATE_MS_PER_CELL;
    return (estimate.cellCount * perCellMs) / 1000;
  }, [computeMode, estimate]);

  const formattedEstimateTime = useMemo(() => {
    if (estimatedSeconds === null) {
      return null;
    }
    return formatDuration(estimatedSeconds);
  }, [estimatedSeconds]);

  const handleParamChange = (field: keyof ParamsState, value: string) => {
    setParams((prev) => ({ ...prev, [field]: value }));
  };

  const handlePresetSelect = (preset: Preset) => {
    setParams({
      observerHeightMeters: preset.observerHeightMeters,
      maxRadiusKm: preset.maxRadiusKm,
      resolutionMeters: preset.resolutionMeters,
    });
  };

  const addObserverPoint = (coords: ObserverState) => {
    setMultiObservers((current) => {
      const exists = current.some(
        (point) => Math.abs(point.lat - coords.lat) < 1e-6 && Math.abs(point.lng - coords.lng) < 1e-6
      );
      if (exists) {
        return current;
      }
      return [...current, coords];
    });
    setMapCenter(coords);
  };

  const applyObserverSelection = (coords: ObserverState) => {
    setMapTool('observer');
    if (isMultiMode) {
      addObserverPoint(coords);
      return;
    }
    setObserver(coords);
    setMapCenter(coords);
  };

  const parseLatLon = (input: string): ObserverState | null => {
    const matches = input.match(/-?\d+(?:\.\d+)?/g);
    if (!matches || matches.length < 2) {
      return null;
    }
    const lat = Number(matches[0]);
    const lon = Number(matches[1]);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return null;
    }
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      return null;
    }
    return { lat, lng: lon };
  };

  const handleSearch = async () => {
    const query = searchQuery.trim();
    if (!query) {
      setSearchStatus('Enter an address or coordinates.');
      return;
    }
    setSearchStatus(null);
    setIsSearching(true);

    const parsed = parseLatLon(query);
    if (parsed) {
      applyObserverSelection(parsed);
      setSearchStatus(null);
      setIsSearching(false);
      return;
    }

    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(query)}`,
        {
          headers: {
            'Accept-Language': 'en',
          },
        }
      );
      if (!response.ok) {
        throw new Error(`Search failed with status ${response.status}`);
      }
      const results = (await response.json()) as Array<{ lat: string; lon: string }>;
      if (!results.length) {
        throw new Error('No results found.');
      }
      const lat = Number(results[0].lat);
      const lon = Number(results[0].lon);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        throw new Error('Invalid search result.');
      }
      applyObserverSelection({ lat, lng: lon });
      setSearchStatus(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unable to search location.';
      setSearchStatus(message);
    } finally {
      setIsSearching(false);
    }
  };

  const normalizeSquareBounds = (start: ObserverState, end: ObserverState): ConsideredBounds => {
    const startPoint = L.CRS.EPSG3857.project(L.latLng(start.lat, start.lng));
    const endPoint = L.CRS.EPSG3857.project(L.latLng(end.lat, end.lng));
    const dx = endPoint.x - startPoint.x;
    const dy = endPoint.y - startPoint.y;
    const size = Math.max(Math.abs(dx), Math.abs(dy));
    const signX = dx >= 0 ? 1 : -1;
    const signY = dy >= 0 ? 1 : -1;
    const squareEnd = L.CRS.EPSG3857.unproject(
      L.point(startPoint.x + signX * size, startPoint.y + signY * size)
    );
    return {
      minLat: Math.min(start.lat, squareEnd.lat),
      minLon: Math.min(start.lng, squareEnd.lng),
      maxLat: Math.max(start.lat, squareEnd.lat),
      maxLon: Math.max(start.lng, squareEnd.lng),
    };
  };

  const handleDrawArea = () => {
    setMapTool('area');
    setAreaDraft(null);
    setStatus('Click two corners to set the considered square.');
  };

  const handleClearArea = () => {
    setConsideredBounds(null);
    setAreaDraft(null);
    if (mapTool === 'area') {
      setMapTool('observer');
      setStatus(null);
    }
  };

  const handleMapTypeChange = (nextIsMulti: boolean) => {
    setIsMultiMode(nextIsMulti);
    if (nextIsMulti) {
      if (observer) {
        setMultiObservers((current) => (current.length === 0 ? [observer] : current));
      }
      return;
    }
    if (multiObservers.length > 0) {
      setObserver(multiObservers[0]);
      setMapCenter(multiObservers[0]);
    }
  };

  const handleUseMyLocation = () => {
    if (!navigator.geolocation) {
      setStatus('Geolocation is not supported in this browser.');
      return;
    }

    setStatus('Requesting location...');
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const coords = {
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        };
        if (isMultiMode) {
          addObserverPoint(coords);
        } else {
          setObserver(coords);
          setMapCenter(coords);
        }
        setStatus(null);
      },
      (error) => {
        setStatus(error.message || 'Unable to fetch current location.');
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  };

  const handleMapSelect = (coords: ObserverState) => {
    if (mapTool === 'area') {
      if (!areaDraft) {
        setAreaDraft(coords);
        setStatus('Now click the opposite corner to finish the area.');
        return;
      }
      const bounds = normalizeSquareBounds(areaDraft, coords);
      setConsideredBounds(bounds);
      setAreaDraft(null);
      setMapTool('observer');
      setStatus(null);
      return;
    }
    applyObserverSelection(coords);
  };

  const handleRemoveObserver = (index: number) => {
    setMultiObservers((current) => current.filter((_, idx) => idx !== index));
  };

  const handleClearObservers = () => {
    setMultiObservers([]);
  };

  const fetchHistory = () => {
    setIsHistoryLoading(true);
    setHistoryError(null);
    fetch(`${API_BASE_URL}/viewshed/history?limit=25`)
      .then(async (response) => {
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || `History request failed with status ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        setHistory(Array.isArray(data.items) ? data.items : []);
      })
      .catch((error: Error) => {
        setHistoryError(error.message || 'Unable to load history.');
      })
      .finally(() => {
        setIsHistoryLoading(false);
      });
  };

  const fetchScenarios = () => {
    setIsScenarioLoading(true);
    setScenarioStatus(null);
    fetch(`${API_BASE_URL}/scenarios`)
      .then(async (response) => {
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || `Scenario request failed with status ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        setScenarios(Array.isArray(data.items) ? data.items : []);
      })
      .catch((error: Error) => {
        setScenarioStatus(error.message || 'Unable to load scenarios.');
      })
      .finally(() => {
        setIsScenarioLoading(false);
      });
  };

  const handleLoadHistory = (item: HistoryItem) => {
    setIsSubmitting(true);
    setSubmitError(null);
    fetch(`${API_BASE_URL}/viewshed/cache/${item.cacheKey}`)
      .then(async (response) => {
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || `Load failed with status ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (data.overlay) {
          setOverlay(data.overlay);
        }
        setLastCacheKey(item.cacheKey ?? null);
        const request = data.request ?? item.request;
        if (request?.observers && request.observers.length > 0) {
          const coordsList = request.observers.map((entry) => ({ lat: entry.lat, lng: entry.lon }));
          setIsMultiMode(true);
          setMultiObservers(coordsList);
          if (coordsList.length > 0) {
            setMapCenter(coordsList[0]);
          }
        } else if (request?.observer) {
          const coords = { lat: request.observer.lat, lng: request.observer.lon };
          setIsMultiMode(false);
          setObserver(coords);
          setMapCenter(coords);
        }
        if (request) {
          setParams((current) => ({
            observerHeightMeters: String(request.observerHeightM ?? current.observerHeightMeters),
            maxRadiusKm: String(request.maxRadiusKm ?? current.maxRadiusKm),
            resolutionMeters: String(request.resolutionM ?? current.resolutionMeters),
          }));
          if (request.mode === 'fast' || request.mode === 'accurate') {
            setComputeMode(request.mode);
          }
          if (typeof request.curvatureEnabled === 'boolean') {
            setCurvatureEnabled(request.curvatureEnabled);
          }
          if (request.consideredBounds) {
            setConsideredBounds(request.consideredBounds);
          } else {
            setConsideredBounds(null);
          }
        }
      })
      .catch((error: Error) => {
        setSubmitError(error.message || 'Unable to load cached viewshed.');
      })
      .finally(() => {
        setIsSubmitting(false);
      });
  };

  const handleDeleteHistory = (item: HistoryItem) => {
    if (!item.cacheKey) {
      return;
    }
    setIsHistoryLoading(true);
    setHistoryError(null);
    fetch(`${API_BASE_URL}/viewshed/cache/${item.cacheKey}`, { method: 'DELETE' })
      .then(async (response) => {
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || `Delete failed with status ${response.status}`);
        }
        return response.json();
      })
      .then(() => {
        setHistory((current) => current.filter((entry) => entry.cacheKey !== item.cacheKey));
      })
      .catch((error: Error) => {
        setHistoryError(error.message || 'Unable to delete cached viewshed.');
      })
      .finally(() => {
        setIsHistoryLoading(false);
      });
  };

  const handleShareOverlay = async () => {
    if (!overlay) {
      return;
    }
    try {
      const binary = atob(overlay.pngBase64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: 'image/png' });
      const file = new File([blob], 'viewshed-overlay.png', { type: 'image/png' });

      if (navigator.canShare && navigator.canShare({ files: [file] })) {
        await navigator.share({
          files: [file],
          title: 'Viewshed Overlay',
        });
        return;
      }

      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'viewshed-overlay.png';
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      setSubmitError('Unable to share overlay.');
    }
  };

  const buildScenarioRequest = (): ScenarioRequest | null => {
    const height = Number(params.observerHeightMeters);
    const radius = Number(params.maxRadiusKm);
    const resolution = Number(params.resolutionMeters);
    if (!Number.isFinite(height) || !Number.isFinite(radius) || !Number.isFinite(resolution)) {
      return null;
    }
    if (isMultiMode) {
      if (multiObservers.length < 2) {
        return null;
      }
      return {
        mapType: 'complex',
        mode: computeMode,
        observers: multiObservers.map((point) => ({ lat: point.lat, lon: point.lng })),
        observerHeightM: height,
        maxRadiusKm: radius,
        resolutionM: resolution,
        consideredBounds: consideredBounds ?? null,
        cacheKey: lastCacheKey ?? null,
        curvatureEnabled,
      };
    }
    if (!observer) {
      return null;
    }
    return {
      mapType: 'single',
      mode: computeMode,
      observer: { lat: observer.lat, lon: observer.lng },
      observerHeightM: height,
      maxRadiusKm: radius,
      resolutionM: resolution,
      consideredBounds: consideredBounds ?? null,
      cacheKey: lastCacheKey ?? null,
      curvatureEnabled,
    };
  };

  const handleSaveScenario = () => {
    const trimmed = scenarioName.trim();
    if (!trimmed) {
      setScenarioStatus('Enter a scenario name.');
      return;
    }
    if (!lastCacheKey) {
      setScenarioStatus('Compute a viewshed before saving the result.');
      return;
    }
    const request = buildScenarioRequest();
    if (!request) {
      setScenarioStatus('Select valid parameters and at least one point before saving.');
      return;
    }
    setScenarioStatus(null);
    setIsScenarioSaving(true);
    fetch(`${API_BASE_URL}/scenarios`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name: trimmed, request }),
    })
      .then(async (response) => {
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || `Save failed with status ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        setScenarios((current) => [data, ...current]);
        setScenarioName('');
      })
      .catch((error: Error) => {
        setScenarioStatus(error.message || 'Unable to save scenario.');
      })
      .finally(() => {
        setIsScenarioSaving(false);
      });
  };

  const handleLoadScenario = (item: ScenarioItem) => {
    const request = item.request;
    if (!request) {
      return;
    }
    if (request.mapType === 'complex' && request.observers && request.observers.length > 0) {
      const coordsList = request.observers.map((entry) => ({ lat: entry.lat, lng: entry.lon }));
      setIsMultiMode(true);
      setMultiObservers(coordsList);
      setMapCenter(coordsList[0]);
    } else if (request.mapType === 'single' && request.observer) {
      const coords = { lat: request.observer.lat, lng: request.observer.lon };
      setIsMultiMode(false);
      setObserver(coords);
      setMapCenter(coords);
    }
    setParams({
      observerHeightMeters: String(request.observerHeightM),
      maxRadiusKm: String(request.maxRadiusKm),
      resolutionMeters: String(request.resolutionM),
    });
    setComputeMode(request.mode);
    setConsideredBounds(request.consideredBounds ?? null);
    setCurvatureEnabled(Boolean(request.curvatureEnabled));
    setAreaDraft(null);
    setMapTool('observer');
    if (request.cacheKey) {
      setIsSubmitting(true);
      setSubmitError(null);
      fetch(`${API_BASE_URL}/viewshed/cache/${request.cacheKey}`)
        .then(async (response) => {
          if (!response.ok) {
            const text = await response.text();
            throw new Error(text || `Load failed with status ${response.status}`);
          }
          return response.json();
        })
        .then((data) => {
          if (data.overlay) {
            setOverlay(data.overlay);
            setLastCacheKey(request.cacheKey ?? null);
          }
        })
        .catch((error: Error) => {
          setScenarioStatus(error.message || 'Unable to load scenario overlay.');
        })
        .finally(() => {
          setIsSubmitting(false);
        });
    } else {
      setOverlay(null);
      setLastCacheKey(null);
    }
  };

  const handleDeleteScenario = (item: ScenarioItem) => {
    fetch(`${API_BASE_URL}/scenarios/${item.id}`, { method: 'DELETE' })
      .then(async (response) => {
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || `Delete failed with status ${response.status}`);
        }
        return response.json();
      })
      .then(() => {
        setScenarios((current) => current.filter((entry) => entry.id !== item.id));
      })
      .catch((error: Error) => {
        setScenarioStatus(error.message || 'Unable to delete scenario.');
      });
  };

  const validateParams = (): FieldErrors => {
    const nextErrors: FieldErrors = {};

    if (isMultiMode) {
      if (multiObservers.length < 2) {
        nextErrors.observers = 'Select at least two points for a complex map.';
      }
    } else if (!observer) {
      nextErrors.observer = 'Pick a location on the map or use your current location.';
    }

    const height = Number(params.observerHeightMeters);
    if (!Number.isFinite(height) || height <= 0) {
      nextErrors.observerHeightMeters = 'Enter a positive height in meters.';
    }

    const radius = Number(params.maxRadiusKm);
    if (!Number.isFinite(radius) || radius <= 0) {
      nextErrors.maxRadiusKm = 'Enter a positive radius in kilometers.';
    }

    const resolution = Number(params.resolutionMeters);
    if (!Number.isFinite(resolution) || resolution <= 0) {
      nextErrors.resolutionMeters = 'Enter a positive resolution in meters.';
    }

    if (estimate && guardrail.blocked) {
      nextErrors.guardrail = `Requested grid ${estimate.gridSide}x${estimate.gridSide} (~${estimate.cellCount.toLocaleString()} cells) exceeds limits.`;
    }

    if (areaDraft) {
      nextErrors.guardrail = 'Finish drawing the considered area (pick the second corner).';
    }

    return nextErrors;
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const nextErrors = validateParams();
    setErrors(nextErrors);

    if (Object.keys(nextErrors).length > 0 || !payloadPreview) {
      return;
    }

    setIsSubmitting(true);
    setSubmitError(null);
    setOverlay(null);
    setLastCacheKey(null);
    setProgress(null);
    setActiveJobId(null);

    if (progressPollRef.current) {
      window.clearInterval(progressPollRef.current);
      progressPollRef.current = null;
    }
    if (requestAbortRef.current) {
      requestAbortRef.current.abort();
    }
    const controller = new AbortController();
    requestAbortRef.current = controller;

    if (isMultiMode) {
      fetch(`${API_BASE_URL}/viewshed/multi/jobs?mode=${computeMode}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payloadPreview),
        signal: controller.signal,
      })
        .then(async (response) => {
          if (!response.ok) {
            const text = await response.text();
            throw new Error(text || `Request failed with status ${response.status}`);
          }
          return response.json();
        })
        .then((data) => {
          requestAbortRef.current = null;
          const jobId = data.jobId as string | undefined;
          const total = Number(data.total ?? multiObservers.length);
          if (!jobId) {
            throw new Error('Failed to start viewshed job.');
          }
          setActiveJobId(jobId);
          setProgress({ completed: 0, total });
          let polling = false;
          progressPollRef.current = window.setInterval(async () => {
            if (polling) {
              return;
            }
            polling = true;
            try {
              const response = await fetch(`${API_BASE_URL}/viewshed/multi/jobs/${jobId}`);
              if (!response.ok) {
                const text = await response.text();
                throw new Error(text || `Progress request failed with status ${response.status}`);
              }
              const job = await response.json();
              if (job.status === 'failed') {
                setSubmitError(job.error || 'Unable to compute viewshed.');
                if (progressPollRef.current) {
                  window.clearInterval(progressPollRef.current);
                  progressPollRef.current = null;
                }
                setIsSubmitting(false);
                setProgress(null);
                setActiveJobId(null);
                return;
              }
              if (job.status === 'canceled') {
                setSubmitError(job.error || 'Canceled.');
                if (progressPollRef.current) {
                  window.clearInterval(progressPollRef.current);
                  progressPollRef.current = null;
                }
                setIsSubmitting(false);
                setProgress(null);
                setActiveJobId(null);
                return;
              }
            if (job.status === 'completed') {
              if (job.result?.overlay) {
                setOverlay(job.result.overlay);
              }
              if (job.result?.cacheKey) {
                setLastCacheKey(job.result.cacheKey);
              } else {
                setLastCacheKey(null);
              }
              if (progressPollRef.current) {
                window.clearInterval(progressPollRef.current);
                progressPollRef.current = null;
              }
              setIsSubmitting(false);
              setProgress(null);
              setActiveJobId(null);
                return;
              }
              const completed = Number(job.completed ?? 0);
              const totalCount = Number(job.total ?? total);
              setProgress({ completed, total: totalCount });
            } catch (error: unknown) {
              const message = error instanceof Error ? error.message : 'Unable to compute viewshed.';
              setSubmitError(message);
              if (progressPollRef.current) {
                window.clearInterval(progressPollRef.current);
                progressPollRef.current = null;
              }
              setIsSubmitting(false);
              setProgress(null);
              setActiveJobId(null);
            } finally {
              polling = false;
            }
          }, 600);
        })
        .catch((error: Error) => {
          if (error.name === 'AbortError') {
            setSubmitError('Canceled.');
            setIsSubmitting(false);
            return;
          }
          setSubmitError(error.message || 'Unable to compute viewshed.');
          setIsSubmitting(false);
        });
      return;
    }

    fetch(`${API_BASE_URL}/viewshed?mode=${computeMode}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payloadPreview),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || `Request failed with status ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        setOverlay(data.overlay ?? null);
        setLastCacheKey(data.cacheKey ?? null);
      })
      .catch((error: Error) => {
        if (error.name === 'AbortError') {
          setSubmitError('Canceled.');
          return;
        }
        setSubmitError(error.message || 'Unable to compute viewshed.');
      })
      .finally(() => {
        setIsSubmitting(false);
        requestAbortRef.current = null;
      });
  };

  const handleCancel = () => {
    if (!isSubmitting) {
      return;
    }
    if (requestAbortRef.current) {
      requestAbortRef.current.abort();
      requestAbortRef.current = null;
    }
    if (progressPollRef.current) {
      window.clearInterval(progressPollRef.current);
      progressPollRef.current = null;
    }
    if (activeJobId) {
      fetch(`${API_BASE_URL}/viewshed/multi/jobs/${activeJobId}/cancel`, { method: 'POST' }).catch(() => {
        // Best-effort cancel.
      });
    }
    setIsSubmitting(false);
    setProgress(null);
    setActiveJobId(null);
    setSubmitError('Canceled.');
  };

  const markers = isMultiMode ? multiObservers : observer ? [observer] : [];
  const consideredBoundsLatLng = consideredBounds
    ? ([
        [consideredBounds.minLat, consideredBounds.minLon],
        [consideredBounds.maxLat, consideredBounds.maxLon],
      ] as [[number, number], [number, number]])
    : null;
  const markerColors = MARKER_COLORS;

  useEffect(() => {
    fetchHistory();
    fetchScenarios();
  }, []);

  useEffect(() => {
    return () => {
      if (progressPollRef.current) {
        window.clearInterval(progressPollRef.current);
        progressPollRef.current = null;
      }
    };
  }, []);

  return (
    <div className={`app${isMapFullscreen ? ' app--map-fullscreen' : ''}`}>
      <header className="app__header">
        <div>
          <h1>Local Viewshed Explorer</h1>
          <p>Select a point to set the observer location.</p>
        </div>
        <button type="button" className="btn" onClick={handleUseMyLocation}>
          Use My Location
        </button>
      </header>

      <section className="panel panel--form">
        <div className="form-layout">
          <form className="form" onSubmit={handleSubmit}>
          <div className="form__group form__group--full">
            <label>Presets</label>
            <div className="presets">
              {PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  className={`preset-btn${matchedPreset?.id === preset.id ? ' preset-btn--active' : ''}`}
                  onClick={() => handlePresetSelect(preset)}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
          <div className="form__group form__group--full">
            <label>Map Type</label>
            <div className="presets">
              <button
                type="button"
                className={`preset-btn${!isMultiMode ? ' preset-btn--active' : ''}`}
                onClick={() => handleMapTypeChange(false)}
              >
                Single
              </button>
              <button
                type="button"
                className={`preset-btn${isMultiMode ? ' preset-btn--active' : ''}`}
                onClick={() => handleMapTypeChange(true)}
              >
                Multi-point
              </button>
            </div>
          </div>
          <div className="form__group form__group--full">
            <label>Calculation Method</label>
            <div className="presets">
              <button
                type="button"
                className={`preset-btn${computeMode === 'fast' ? ' preset-btn--active' : ''}`}
                onClick={() => setComputeMode('fast')}
              >
                Default
              </button>
              <button
                type="button"
                className={`preset-btn${computeMode === 'accurate' ? ' preset-btn--active' : ''}`}
                onClick={() => setComputeMode('accurate')}
              >
                Advanced
              </button>
            </div>
          </div>
          <div className="form__group form__group--full">
            <label>Earth Curvature</label>
            <div className="presets">
              <button
                type="button"
                className={`preset-btn${!curvatureEnabled ? ' preset-btn--active' : ''}`}
                onClick={() => setCurvatureEnabled(false)}
              >
                Off
              </button>
              <button
                type="button"
                className={`preset-btn${curvatureEnabled ? ' preset-btn--active' : ''}`}
                onClick={() => setCurvatureEnabled(true)}
              >
                On
              </button>
            </div>
          </div>
          <div className="form__group">
            <label htmlFor="observerHeightMeters">Observer Height (meters)</label>
            <input
              id="observerHeightMeters"
              name="observerHeightMeters"
              type="number"
              inputMode="decimal"
              min="0"
              step="0.1"
              value={params.observerHeightMeters}
              onChange={(event) => handleParamChange('observerHeightMeters', event.target.value)}
            />
            {errors.observerHeightMeters ? <div className="error">{errors.observerHeightMeters}</div> : null}
          </div>
          <div className="form__group">
            <label htmlFor="maxRadiusKm">Max Radius (km)</label>
            <input
              id="maxRadiusKm"
              name="maxRadiusKm"
              type="number"
              inputMode="decimal"
              min="0"
              step="0.5"
              value={params.maxRadiusKm}
              onChange={(event) => handleParamChange('maxRadiusKm', event.target.value)}
            />
            {errors.maxRadiusKm ? <div className="error">{errors.maxRadiusKm}</div> : null}
          </div>
          <div className="form__group">
            <label htmlFor="resolutionMeters">Resolution (meters)</label>
            <input
              id="resolutionMeters"
              name="resolutionMeters"
              type="number"
              inputMode="decimal"
              min="0"
              step="1"
              value={params.resolutionMeters}
              onChange={(event) => handleParamChange('resolutionMeters', event.target.value)}
            />
            {errors.resolutionMeters ? <div className="error">{errors.resolutionMeters}</div> : null}
          </div>
          <div className="form__group form__group--full">
            <label>Considered Area</label>
            <div className="presets">
              <button
                type="button"
                className={`preset-btn${mapTool === 'area' ? ' preset-btn--active' : ''}`}
                onClick={handleDrawArea}
              >
                {mapTool === 'area' ? 'Click 2 Corners' : 'Draw Area'}
              </button>
              <button
                type="button"
                className="preset-btn"
                onClick={handleClearArea}
                disabled={!consideredBounds && !areaDraft}
              >
                Clear Area
              </button>
            </div>
            {consideredBounds ? (
              <div className="estimate">
                Area: {consideredBounds.minLat.toFixed(3)}, {consideredBounds.minLon.toFixed(3)} →{' '}
                {consideredBounds.maxLat.toFixed(3)}, {consideredBounds.maxLon.toFixed(3)}
              </div>
            ) : null}
            {areaDraft ? <div className="warning">Click the opposite corner to finish the square.</div> : null}
          </div>
          <div className="form__group form__group--full">
            <label>Estimate</label>
            <div className="estimate">
              {estimate
                ? `Grid ${estimate.gridWidth}x${estimate.gridHeight} (~${estimate.cellCount.toLocaleString()} cells)`
                : 'Enter radius and resolution to estimate grid size.'}
            </div>
            {estimate && formattedEstimateTime ? (
              <div className="estimate estimate--time">
                Est. compute time ({computeMode === 'fast' ? 'Default' : 'Advanced'}): ~{formattedEstimateTime}{' '}
                <span className="estimate__note">(excludes DEM fetch/cache)</span>
              </div>
            ) : null}
            {guardrail.warnings.map((warning) => (
              <div key={warning} className="warning">
                {warning}
              </div>
            ))}
            {errors.guardrail ? <div className="error">{errors.guardrail}</div> : null}
          </div>
          <div className="form__actions">
            <button className="btn" type="submit" disabled={isSubmitting || guardrail.blocked}>
              {isSubmitting ? 'Submitting...' : isMultiMode ? 'Compute Complex Map' : 'Compute Viewshed'}
            </button>
            <button className="btn btn--ghost" type="button" onClick={handleCancel} disabled={!isSubmitting}>
              Cancel
            </button>
            <button
              className="btn btn--ghost"
              type="button"
              onClick={() => {
                setOverlay(null);
                setLastCacheKey(null);
              }}
              disabled={!overlay}
            >
              Clear Overlay
            </button>
            <button className="btn btn--ghost" type="button" onClick={handleShareOverlay} disabled={!overlay}>
              Share Overlay
            </button>
          </div>
          {isSubmitting ? (
            <div className="loading">
              <div className="loading__bar" />
              <span className="loading__label">
                {progress ? `Computing viewshed… (${progress.completed}/${progress.total})` : 'Computing viewshed…'}
              </span>
            </div>
          ) : null}
          {submitError ? <div className="error form__error">{submitError}</div> : null}
          </form>
          <aside className="positions">
            <div className="search-panel">
              <label htmlFor="locationSearch">Search Address or Coordinates</label>
              <div className="search">
                <input
                  id="locationSearch"
                  name="locationSearch"
                  type="text"
                  placeholder="e.g. 40.2338, -111.6585 or Provo, UT"
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault();
                      void handleSearch();
                    }
                  }}
                />
                <button className="btn btn--ghost" type="button" onClick={handleSearch} disabled={isSearching}>
                  {isSearching ? 'Searching…' : 'Search'}
                </button>
              </div>
              {searchStatus ? <div className="status">{searchStatus}</div> : null}
            </div>
            <h2>Positions</h2>
            {errors.observers ? <div className="status">{errors.observers}</div> : null}
            {errors.observer ? <div className="status">{errors.observer}</div> : null}
            {status ? <div className="status">{status}</div> : null}
            {isMultiMode ? (
              <div className="points">
                {multiObservers.length === 0 ? (
                  <div className="status">No points yet. Click the map to add observer points.</div>
                ) : (
                  <div className="points__list">
                    {multiObservers.map((point, index) => (
                      <div key={`${point.lat}-${point.lng}-${index}`} className="points__item">
                        <span className="points__meta">
                          <span
                            className="points__color"
                            style={{ backgroundColor: markerColors[index % markerColors.length] }}
                          />
                          {point.lat.toFixed(5)}, {point.lng.toFixed(5)}
                        </span>
                        <button
                          type="button"
                          className="points__remove"
                          onClick={() => handleRemoveObserver(index)}
                        >
                          Remove
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <div className="points__actions">
                  <button
                    type="button"
                    className="btn btn--ghost"
                    onClick={handleClearObservers}
                    disabled={multiObservers.length === 0}
                  >
                    Clear Points
                  </button>
                </div>
              </div>
            ) : observer ? (
              <div className="points">
                <div className="points__item">
                  <span className="points__meta">
                    <span className="points__color" style={{ backgroundColor: markerColors[0] }} />
                    {observer.lat.toFixed(5)}, {observer.lng.toFixed(5)}
                  </span>
                </div>
              </div>
            ) : (
              <div className="status">Select a point on the map to set the observer.</div>
            )}
          </aside>
        </div>
      </section>

      <section className="panel panel--columns">
        <div className="history">
          <div className="history__header">
            <h2>Recent Viewsheds</h2>
            <div className="history__actions">
              <button
                className="btn btn--ghost"
                type="button"
                onClick={() => setIsHistoryCollapsed((current) => !current)}
              >
                {isHistoryCollapsed ? 'Show' : 'Hide'}
              </button>
              <button className="btn btn--ghost" type="button" onClick={fetchHistory} disabled={isHistoryLoading}>
                {isHistoryLoading ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>
          </div>
          {!isHistoryCollapsed ? (
            <>
              {historyError ? <div className="error">{historyError}</div> : null}
              {history.length === 0 && !historyError ? (
                <div className="status">No cached viewsheds yet.</div>
              ) : (
                <ul className="history__list">
                  {history.map((item) => {
                    const createdAt = item.createdAt ? new Date(item.createdAt).toLocaleString() : 'Unknown time';
                    const observers = item.request?.observers ?? null;
                    const lat = observers && observers.length > 0 ? observers[0].lat : item.request?.observer?.lat;
                    const lon = observers && observers.length > 0 ? observers[0].lon : item.request?.observer?.lon;
                    const radius = item.request?.maxRadiusKm;
                    const resolution = item.request?.resolutionM;
                    const mode = item.request?.mode;
                    const isMulti = Boolean(observers && observers.length > 0);
                    return (
                      <li key={item.cacheKey} className="history__item">
                        <button className="history__button" type="button" onClick={() => handleLoadHistory(item)}>
                          <div className="history__title">{createdAt}</div>
                          <div className="history__meta">
                            {isMulti
                              ? `Multi (${observers?.length ?? 0} points)`
                              : lat !== undefined && lon !== undefined
                              ? `${lat.toFixed(3)}, ${lon.toFixed(3)}`
                              : 'Unknown location'}
                          </div>
                          <div className="history__meta">
                            {radius !== undefined && resolution !== undefined
                              ? `${radius} km · ${resolution} m`
                              : 'Unknown params'}
                          </div>
                          {mode ? (
                            <div className="history__meta">{mode === 'fast' ? 'Default' : 'Advanced'}</div>
                          ) : null}
                        </button>
                        <button
                          className="history__delete"
                          type="button"
                          onClick={() => handleDeleteHistory(item)}
                          aria-label="Delete viewshed"
                        >
                          Delete
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </>
          ) : null}
        </div>
        <div className="history scenarios-panel">
          <div className="history__header">
            <h2>Saved Viewsheds</h2>
            <div className="history__actions">
              <button
                className="btn btn--ghost"
                type="button"
                onClick={() => setIsScenarioCollapsed((current) => !current)}
              >
                {isScenarioCollapsed ? 'Show' : 'Hide'}
              </button>
              <button
                className="btn btn--ghost"
                type="button"
                onClick={fetchScenarios}
                disabled={isScenarioLoading}
              >
                {isScenarioLoading ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>
          </div>
          <div className="scenario-save">
            <label htmlFor="scenarioName">Scenario Name</label>
            <div className="search">
              <input
                id="scenarioName"
                name="scenarioName"
                type="text"
                placeholder="e.g. Logan overview"
                value={scenarioName}
                onChange={(event) => setScenarioName(event.target.value)}
              />
              <button className="btn btn--ghost" type="button" onClick={handleSaveScenario} disabled={isScenarioSaving}>
                {isScenarioSaving ? 'Saving…' : 'Save Scenario'}
              </button>
            </div>
            {scenarioStatus ? <div className="status">{scenarioStatus}</div> : null}
          </div>
          {!isScenarioCollapsed ? (
            <>
              {scenarios.length === 0 ? (
                <div className="status">No saved scenarios yet.</div>
              ) : (
                <ul className="scenarios__list">
                  {scenarios.map((item) => (
                    <li key={item.id} className="scenarios__item">
                      <button className="scenarios__button" type="button" onClick={() => handleLoadScenario(item)}>
                        <div className="scenarios__title">{item.name}</div>
                        <div className="scenarios__meta">{new Date(item.createdAt).toLocaleString()}</div>
                      </button>
                      <button
                        className="scenarios__delete"
                        type="button"
                        onClick={() => handleDeleteScenario(item)}
                      >
                        Delete
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </>
          ) : null}
        </div>
      </section>

      <section className={`map${isMapFullscreen ? ' map--fullscreen' : ''}`}>
        <MapContainer center={mapCenter} zoom={DEFAULT_ZOOM} scrollWheelZoom className="map__container">
          <MapViewController center={mapCenter} />
          <MapClickHandler onSelect={handleMapSelect} />
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {markers.map((point, index) => (
            <Marker
              key={`${point.lat}-${point.lng}-${index}`}
              position={point}
              icon={getColoredMarkerIcon(markerColors[index % markerColors.length])}
            />
          ))}
          {consideredBoundsLatLng ? (
            <Rectangle bounds={consideredBoundsLatLng} pathOptions={{ color: '#0f172a', weight: 2, dashArray: '4' }} />
          ) : null}
          {overlay ? (
            <ImageOverlay
              url={`data:image/png;base64,${overlay.pngBase64}`}
              bounds={[
                [overlay.boundsLatLon[0], overlay.boundsLatLon[1]],
                [overlay.boundsLatLon[2], overlay.boundsLatLon[3]],
              ]}
              opacity={1}
            />
          ) : null}
        </MapContainer>
        <div className="map__fullscreen">
          <button className="btn btn--ghost" type="button" onClick={() => setIsMapFullscreen((value) => !value)}>
            {isMapFullscreen ? 'Exit Full Screen' : 'Full Screen'}
          </button>
        </div>
      </section>
    </div>
  );
}
