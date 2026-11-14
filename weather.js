/**
 * ======================================================
 * AGRI-WEATHER & MICROCLIMATE STATION (weather.js)
 * ======================================================
 * Integrates OpenWeatherMap API with crop disease pathology
 * to assess fungal/bacterial outbreak risks based on live
 * humidity, temperature, and wind conditions.
 */

// OpenWeatherMap API Configuration
const WEATHER_API_KEY = '056e11db575389020b59170ab58e29c1';

// Weather Condition Dynamic Photography / Artwork
const WEATHER_PHOTOS = {
    Clear: "https://images.unsplash.com/photo-1601297183305-6df142704ea2?auto=format&fit=crop&w=800&q=80",
    Clouds: "https://images.unsplash.com/photo-1534088568595-a066f410bcda?auto=format&fit=crop&w=800&q=80",
    Rain: "https://images.unsplash.com/photo-1519692933481-e162a57d6721?auto=format&fit=crop&w=800&q=80",
    Drizzle: "https://images.unsplash.com/photo-1556484687-30636164638a?auto=format&fit=crop&w=800&q=80",
    Thunderstorm: "https://images.unsplash.com/photo-1605727216801-e27ce1d0cc28?auto=format&fit=crop&w=800&q=80",
    Snow: "https://images.unsplash.com/photo-1517299321909-504289274659?auto=format&fit=crop&w=800&q=80",
    Mist: "https://images.unsplash.com/photo-1487621167305-5d248087c724?auto=format&fit=crop&w=800&q=80",
    Haze: "https://images.unsplash.com/photo-1522163182402-834f871fd851?auto=format&fit=crop&w=800&q=80"
};

/**
 * Calculates crop disease vulnerability index from atmospheric metrics
 */
function evaluateAgriRisk(temp, humidity, windSpeed, condition) {
    let riskLevel = "LOW";
    let riskColor = "#22c55e";
    let advice = "Optimal foliar conditions. Standard surveillance recommended.";
    let sprayCondition = "Favorable for pesticide/fungicide application.";

    // Fungal pathogens (Early Blight, Rust, Septoria) thrive in high humidity (>75%) & moderate temps (18-28C)
    if (humidity >= 80) {
        riskLevel = "CRITICAL";
        riskColor = "#ef4444";
        advice = "Extreme foliar humidity! High susceptibility to Leaf Blight, Rust, and Septoria spore propagation.";
    } else if (humidity >= 65) {
        riskLevel = "ELEVATED";
        riskColor = "#f97316";
        advice = "Elevated moisture detected. Inspect lower leaf canopy for early fungal lesions.";
    } else if (temp > 35) {
        riskLevel = "HEAT STRESS";
        riskColor = "#facc15";
        advice = "High thermal exposure may exacerbate foliar water loss. Ensure adequate irrigation.";
    }

    if (windSpeed > 6.0) {
        sprayCondition = "High wind drift warning (>6 m/s). Avoid spraying chemicals to prevent aerosol loss.";
    } else if (condition.toLowerCase().includes("rain")) {
        sprayCondition = "Precipitation alert. Delay foliar spray as rainfall will wash away protective treatments.";
    }

    return { riskLevel, riskColor, advice, sprayCondition };
}

/**
 * Fetch and render live weather telemetry
 */
async function fetchWeatherData(searchCity) {
    const cityInput = document.getElementById('city');
    const city = searchCity || (cityInput ? cityInput.value.trim() : '');
    const weatherContainer = document.getElementById('weather');

    if (!weatherContainer) return;

    if (!city) {
        weatherContainer.innerHTML = `
            <div class="weather-alert weather-alert-warning">
                <span>⚠️</span>
                <div>
                    <strong>Location Required</strong>
                    <p style="margin: 3px 0 0; font-size: 0.85rem;">Please enter a town or city name to retrieve microclimate data.</p>
                </div>
            </div>`;
        return;
    }

    // 1. RECTIFICATION: Show animated loading indicator BEFORE network dispatch
    weatherContainer.innerHTML = `
        <div class="weather-loading-card">
            <div class="weather-spinner"></div>
            <p>Gathering satellite microclimate telemetry for <strong>${city}</strong>...</p>
        </div>`;

    try {
        const endpoint = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(city)}&appid=${WEATHER_API_KEY}&units=metric`;
        const response = await fetch(endpoint);
        const data = await response.json();

        if (response.ok) {
            renderAttractiveWeather(data, weatherContainer);
        } else {
            weatherContainer.innerHTML = `
                <div class="weather-alert weather-alert-error">
                    <span>❌</span>
                    <div>
                        <strong>City "${city}" Not Found</strong>
                        <p style="margin: 3px 0 0; font-size: 0.85rem;">${data.message || 'Please check spelling (e.g., "Delhi", "London", "Nairobi") and try again.'}</p>
                    </div>
                </div>`;
        }
    } catch (error) {
        weatherContainer.innerHTML = `
            <div class="weather-alert weather-alert-error">
                <span>📡</span>
                <div>
                    <strong>Network Communication Error</strong>
                    <p style="margin: 3px 0 0; font-size: 0.85rem;">Unable to contact weather satellites. Please verify your internet connection.</p>
                </div>
            </div>`;
    }
}

/**
 * Renders an attractive visual dashboard with weather photography,
 * condition badges, and agricultural disease risk indexing.
 */
function renderAttractiveWeather(data, container) {
    const { main, weather, wind, visibility, sys, name } = data;
    const condition = weather[0].main;
    const desc = weather[0].description;
    const iconCode = weather[0].icon;
    const country = sys ? sys.country : '';

    // Selected picture corresponding to condition
    const photoUrl = WEATHER_PHOTOS[condition] || WEATHER_PHOTOS.Clear;
    const iconUrl = `https://openweathermap.org/img/wn/${iconCode}@4x.png`;

    // Calculate crop pathology risk
    const agri = evaluateAgriRisk(main.temp, main.humidity, wind.speed, desc);

    container.innerHTML = `
        <div class="weather-dashboard-card">
            <!-- Hero Weather Banner with Picture -->
            <div class="weather-hero" style="background-image: linear-gradient(to right, rgba(7, 21, 47, 0.92) 20%, rgba(16, 38, 74, 0.7)), url('${photoUrl}');">
                <div class="weather-hero-content">
                    <div class="weather-location-pill">📍 ${name}${country ? ', ' + country : ''}</div>
                    <div class="weather-temp-display">
                        <span class="weather-temp-val">${Math.round(main.temp)}°C</span>
                        <div class="weather-desc-group">
                            <span class="weather-condition-text">${desc.toUpperCase()}</span>
                            <span class="weather-feels-like">Feels like ${Math.round(main.feels_like)}°C • High: ${Math.round(main.temp_max)}°C / Low: ${Math.round(main.temp_min)}°C</span>
                        </div>
                    </div>
                </div>
                <div class="weather-hero-illustration">
                    <img src="${iconUrl}" alt="${desc}" class="weather-big-icon">
                </div>
            </div>

            <!-- Agricultural Disease Risk Meter -->
            <div class="agri-risk-bar" style="border-left-color: ${agri.riskColor};">
                <div class="agri-risk-header">
                    <span style="font-size: 0.82rem; letter-spacing: 0.5px; text-transform: uppercase; color: #94a3b8;">Foliar Pathogen Vulnerability:</span>
                    <span class="agri-risk-badge" style="background-color: ${agri.riskColor}22; color: ${agri.riskColor}; border: 1px solid ${agri.riskColor}55;">
                        ● ${agri.riskLevel} RISK
                    </span>
                </div>
                <p class="agri-risk-text">${agri.advice}</p>
                <p class="agri-spray-text">🚜 <strong>Spraying Advisory:</strong> ${agri.sprayCondition}</p>
            </div>

            <!-- Meteorological Metrics Grid -->
            <div class="weather-grid">
                <div class="weather-metric-tile">
                    <div class="metric-icon">💧</div>
                    <div class="metric-info">
                        <span class="metric-label">Relative Humidity</span>
                        <span class="metric-value">${main.humidity}%</span>
                        <span class="metric-sub">${main.humidity > 70 ? 'High Disease Spread' : 'Moderate Moisture'}</span>
                    </div>
                </div>

                <div class="weather-metric-tile">
                    <div class="metric-icon">💨</div>
                    <div class="metric-info">
                        <span class="metric-label">Wind Velocity</span>
                        <span class="metric-value">${wind.speed} m/s</span>
                        <span class="metric-sub">${(wind.speed * 3.6).toFixed(1)} km/h</span>
                    </div>
                </div>

                <div class="weather-metric-tile">
                    <div class="metric-icon">👁️</div>
                    <div class="metric-info">
                        <span class="metric-label">Atmospheric Clarity</span>
                        <span class="metric-value">${(visibility / 1000).toFixed(1)} km</span>
                        <span class="metric-sub">${visibility >= 8000 ? 'Clear Canopy Visibility' : 'Hazy / Humid'}</span>
                    </div>
                </div>

                <div class="weather-metric-tile">
                    <div class="metric-icon">🌡️</div>
                    <div class="metric-info">
                        <span class="metric-label">Barometric Pressure</span>
                        <span class="metric-value">${main.pressure} hPa</span>
                        <span class="metric-sub">Sea-Level Pressure</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * Setup event listeners and quick preset actions
 */
function initWeatherModule() {
    // 1. Support user's snippet id 'predict' (if used as weather button)
    const predictBtn = document.getElementById('predict');
    if (predictBtn) {
        predictBtn.addEventListener('click', () => fetchWeatherData());
    }

    // 2. Support dedicated weather search button
    const searchBtn = document.getElementById('btn-search-weather');
    if (searchBtn) {
        searchBtn.addEventListener('click', () => fetchWeatherData());
    }

    // 3. Support Enter key inside city input
    const cityInput = document.getElementById('city');
    if (cityInput) {
        cityInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                fetchWeatherData();
            }
        });
    }

    // 4. Quick preset chips
    const chips = document.querySelectorAll('.weather-quick-chip');
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            const cityName = chip.getAttribute('data-city');
            if (cityInput) cityInput.value = cityName;
            fetchWeatherData(cityName);
        });
    });

    // 5. Geolocation support
    const geoBtn = document.getElementById('btn-weather-geo');
    if (geoBtn) {
        geoBtn.addEventListener('click', () => {
            if (!navigator.geolocation) {
                alert("Geolocation is not supported by your browser.");
                return;
            }
            const weatherContainer = document.getElementById('weather');
            if (weatherContainer) {
                weatherContainer.innerHTML = `
                    <div class="weather-loading-card">
                        <div class="weather-spinner"></div>
                        <p>Accessing GPS coordinates for your field...</p>
                    </div>`;
            }
            navigator.geolocation.getCurrentPosition(async (pos) => {
                const lat = pos.coords.latitude;
                const lon = pos.coords.longitude;
                try {
                    const endpoint = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${WEATHER_API_KEY}&units=metric`;
                    const res = await fetch(endpoint);
                    const data = await res.json();
                    if (res.ok) {
                        if (cityInput) cityInput.value = data.name;
                        renderAttractiveWeather(data, weatherContainer);
                    }
                } catch (e) {
                    fetchWeatherData("Delhi");
                }
            }, () => {
                fetchWeatherData("Delhi");
            });
        });
    }

    // Load initial weather for a prominent farming hub
    fetchWeatherData("Delhi");
}

// Initialize on DOM load
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initWeatherModule);
} else {
    initWeatherModule();
}
