import { useEffect, useMemo, useState, type FormEvent } from 'react';
import type { LatLngLiteral } from 'leaflet';
import L from 'leaflet';
import { ImageOverlay, MapContainer, Marker, TileLayer, useMap, useMapEvents } from 'react-leaflet';
import iconRetinaUrl from 'leaflet/dist/images/marker-icon-2x.png';
import iconUrl from 'leaflet/dist/images/marker-icon.png';
import shadowUrl from 'leaflet/dist/images/marker-shadow.png';

const DEFAULT_MARKER_ICON = L.icon({
  iconRetinaUrl,
  iconUrl,
  shadowUrl,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

const DEFAULT_CENTER: LatLngLiteral = { lat: 20, lng: 0 };
// Approx. 25 miles across on a typical laptop viewport.
const DEFAULT_ZOOM = 11;
const API_BASE_URL = 'http://localhost:8000';
const WARN_CELL_COUNT = 1_000_000;
const MAX_CELL_COUNT = 4_000_000;
const MAX_GRID_SIDE = 2000;

type ObserverState = {
  lat: number;
  lng: number;
};

type ParamsState = {
  observerHeightMeters: string;
  maxRadiusKm: string;
  resolutionMeters: string;
};

type FieldErrors = Partial<Record<keyof ParamsState | 'observer' | 'guardrail', string>>;

type OverlayPayload = {
  pngBase64: string;
  boundsLatLon: [number, number, number, number];
};

type HistoryItem = {
  cacheKey: string;
  createdAt?: string | null;
  demVersion?: string | null;
  request?: {
    observer?: {
      lat: number;
      lon: number;
    };
    observerHeightM?: number;
    maxRadiusKm?: number;
    resolutionM?: number;
  } | null;
  boundsLatLon?: [number, number, number, number] | null;
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
    label: 'Fast',
    observerHeightMeters: '1.7',
    maxRadiusKm: '10',
    resolutionMeters: '90',
  },
  {
    id: 'medium',
    label: 'Medium',
    observerHeightMeters: '1.7',
    maxRadiusKm: '25',
    resolutionMeters: '60',
  },
  {
    id: 'high',
    label: 'High',
    observerHeightMeters: '1.7',
    maxRadiusKm: '50',
    resolutionMeters: '30',
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
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [isHistoryCollapsed, setIsHistoryCollapsed] = useState(false);

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

  const payloadPreview = useMemo(() => {
    if (!observerForApi) {
      return null;
    }

    return {
      observer: observerForApi,
      observerHeightM: Number(params.observerHeightMeters),
      maxRadiusKm: Number(params.maxRadiusKm),
      resolutionM: Number(params.resolutionMeters),
    };
  }, [observerForApi, params]);

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
    const gridSide = Math.ceil((2 * radiusM) / resolutionM) + 1;
    const cellCount = gridSide * gridSide;
    return { gridSide, cellCount };
  }, [params.maxRadiusKm, params.resolutionMeters]);

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
        setObserver(coords);
        setMapCenter(coords);
        setStatus(null);
      },
      (error) => {
        setStatus(error.message || 'Unable to fetch current location.');
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
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
        const request = data.request ?? item.request;
        if (request?.observer) {
          const coords = { lat: request.observer.lat, lng: request.observer.lon };
          setObserver(coords);
          setMapCenter(coords);
        }
        if (request) {
          setParams((current) => ({
            observerHeightMeters: String(request.observerHeightM ?? current.observerHeightMeters),
            maxRadiusKm: String(request.maxRadiusKm ?? current.maxRadiusKm),
            resolutionMeters: String(request.resolutionM ?? current.resolutionMeters),
          }));
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

  const validateParams = (): FieldErrors => {
    const nextErrors: FieldErrors = {};

    if (!observer) {
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

    fetch(`${API_BASE_URL}/viewshed`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payloadPreview),
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
      })
      .catch((error: Error) => {
        setSubmitError(error.message || 'Unable to compute viewshed.');
      })
      .finally(() => {
        setIsSubmitting(false);
      });
  };

  const observerText = observer
    ? `${observer.lat.toFixed(6)}, ${observer.lng.toFixed(6)}`
    : 'Not set';

  useEffect(() => {
    fetchHistory();
  }, []);

  return (
    <div className="app">
      <header className="app__header">
        <div>
          <h1>Local Viewshed Explorer</h1>
          <p>Select a point to set the observer location.</p>
        </div>
        <button type="button" className="btn" onClick={handleUseMyLocation}>
          Use My Location
        </button>
      </header>

      <section className="panel">
        <div>
          <h2>Observer</h2>
          <div className="panel__row">
            <span className="label">Lat/Lon</span>
            <span className="value">{observerText}</span>
          </div>
          {errors.observer ? <div className="status">{errors.observer}</div> : null}
          {status ? <div className="status">{status}</div> : null}
        </div>
        <div>
          <h2>API Payload</h2>
          <pre className="code">
{payloadPreview ? JSON.stringify(payloadPreview, null, 2) : '// Awaiting observer location'}
          </pre>
        </div>
      </section>

      <section className="panel">
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
            <label>Estimate</label>
            <div className="estimate">
              {estimate
                ? `Grid ${estimate.gridSide}x${estimate.gridSide} (~${estimate.cellCount.toLocaleString()} cells)`
                : 'Enter radius and resolution to estimate grid size.'}
            </div>
            {guardrail.warnings.map((warning) => (
              <div key={warning} className="warning">
                {warning}
              </div>
            ))}
            {errors.guardrail ? <div className="error">{errors.guardrail}</div> : null}
          </div>
          <div className="form__actions">
            <button className="btn" type="submit" disabled={isSubmitting || guardrail.blocked}>
              {isSubmitting ? 'Submitting...' : 'Compute Viewshed'}
            </button>
            <button
              className="btn btn--ghost"
              type="button"
              onClick={() => setOverlay(null)}
              disabled={!overlay}
            >
              Clear Overlay
            </button>
          </div>
          {isSubmitting ? (
            <div className="loading">
              <div className="loading__bar" />
              <span className="loading__label">Computing viewshed…</span>
            </div>
          ) : null}
          {submitError ? <div className="error form__error">{submitError}</div> : null}
        </form>
      </section>

      <section className="panel">
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
                    const lat = item.request?.observer?.lat;
                    const lon = item.request?.observer?.lon;
                    const radius = item.request?.maxRadiusKm;
                    const resolution = item.request?.resolutionM;
                    return (
                      <li key={item.cacheKey} className="history__item">
                        <button className="history__button" type="button" onClick={() => handleLoadHistory(item)}>
                          <div className="history__title">{createdAt}</div>
                          <div className="history__meta">
                            {lat !== undefined && lon !== undefined
                              ? `${lat.toFixed(3)}, ${lon.toFixed(3)}`
                              : 'Unknown location'}
                          </div>
                          <div className="history__meta">
                            {radius !== undefined && resolution !== undefined
                              ? `${radius} km · ${resolution} m`
                              : 'Unknown params'}
                          </div>
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
      </section>

      <section className="map">
        <MapContainer center={mapCenter} zoom={DEFAULT_ZOOM} scrollWheelZoom className="map__container">
          <MapViewController center={mapCenter} />
          <MapClickHandler onSelect={(coords) => setObserver(coords)} />
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {observer ? <Marker position={observer} icon={DEFAULT_MARKER_ICON} /> : null}
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
      </section>
    </div>
  );
}
