param(
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

Write-Host "Testing API at $BaseUrl" -ForegroundColor Cyan

# Health
Write-Host "GET /health" -ForegroundColor Yellow
$health = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get
$health | ConvertTo-Json -Depth 5

# Info
Write-Host "GET /info" -ForegroundColor Yellow
$info = Invoke-RestMethod -Uri "$BaseUrl/info" -Method Get
$info | ConvertTo-Json -Depth 5

# Predict
Write-Host "POST /predict" -ForegroundColor Yellow
$body = @{
    hour_sin = 0.5
    hour_cos = 0.866
    dow_sin = 0.0
    dow_cos = 1.0
    month_sin = 0.0
    month_cos = 1.0
    lag_1h = 15.0
    lag_24h = 20.0
    lag_168h = 18.0
    diff_24h = -5.0
    rolling_7d_mean = 17.5
    rolling_7d_std = 3.2
    rolling_14d_mean = 16.8
    rolling_7d_cv = 0.18
    zone_mean_demand = 22.0
    zone_rank = 25
    zone_is_top50 = 1
} | ConvertTo-Json

$pred = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method Post -ContentType "application/json" -Body $body
$pred | ConvertTo-Json -Depth 5

Write-Host "Smoke test complete." -ForegroundColor Green
