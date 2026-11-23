# Final evaluation run for the paper (RTX 5080 machine, Ollama installed).
# Every stage is resumable: re-running the script continues from the saved JSON files.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_final_5080.ps1 -Stage heldout
#   powershell -ExecutionPolicy Bypass -File scripts\run_final_5080.ps1 -Stage t17
#   powershell -ExecutionPolicy Bypass -File scripts\run_final_5080.ps1 -Stage crisis
#   powershell -ExecutionPolicy Bypass -File scripts\run_final_5080.ps1 -Stage audit
#   powershell -ExecutionPolicy Bypass -File scripts\run_final_5080.ps1 -Stage tables
#
# Do NOT edit code, prompts, or configs between stages: the held-out suite is only valid if the
# system is frozen before it is run. If you must change something, rerun everything.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("check", "heldout", "t17", "crisis", "audit", "tables")]
    [string]$Stage
)

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Create the venv first (see docs/RUN_ON_5080.md)." }
New-Item -ItemType Directory -Force outputs\final | Out-Null

# Keep both models resident in VRAM (7B ~5 GB + 14B ~9 GB at Q4_K_M).
$env:OLLAMA_MAX_LOADED_MODELS = "2"
$env:REVITLS_DISABLE_INNER_TQDM = "0"

function Assert-Clean {
    $dirty = git status --porcelain
    if ($dirty) {
        Write-Host "Working tree is not clean:`n$dirty" -ForegroundColor Yellow
        throw "Commit or stash everything first so env.json records the exact code that produced the results."
    }
}

switch ($Stage) {
    "check" {
        & $py -m pytest -q
        ollama list
        & $py scripts\run_heldout_suite.py --dry-run --limit-cases 12 --output outputs\dry\smoke.json --fresh
        Write-Host "Smoke test OK" -ForegroundColor Green
    }
    "heldout" {
        Assert-Clean
        & $py scripts\env_report.py | Out-Null
        & $py scripts\run_heldout_suite.py --config configs\hybrid_revitls.yaml --output outputs\final\heldout.json
        & $py scripts\make_paper_tables.py
    }
    "t17" {
        Assert-Clean
        & $py scripts\env_report.py | Out-Null
        & $py scripts\run_fullcorpus.py --datasets t17 --ablations --output outputs\final\summary.json
        & $py scripts\make_paper_tables.py
    }
    "crisis" {
        Assert-Clean
        & $py scripts\run_fullcorpus.py --datasets crisis --output outputs\final\summary.json
        & $py scripts\make_paper_tables.py
    }
    "audit" {
        & $py scripts\export_transition_audit.py --export-random-audit --audit-per-stratum 5 --audit-seed 20261123
        Write-Host "Now annotate outputs\final\transition_audit_random.csv (two annotators, independently), then run:" -ForegroundColor Cyan
        Write-Host "  .\.venv\Scripts\python.exe scripts\summarize_random_audit.py"
    }
    "tables" {
        & $py scripts\make_paper_tables.py
        Write-Host "Tables written to paper\generated" -ForegroundColor Green
    }
}
