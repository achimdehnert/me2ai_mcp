# ME2AI PyPI Package Update and Upload Workflow
# This script automates the process of updating and uploading packages to PyPI

param (
    [Parameter(Mandatory=$true)]
    [string]$NewVersion,
    
    [Parameter(Mandatory=$false)]
    [string]$EnvFilePath = "$HOME\github\me2ai\.env",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipTests = $false
)

# Ensure we're in the package root directory
$PackageRoot = Get-Location
$PackageName = Split-Path -Leaf $PackageRoot

# 1. Verify dependencies are installed
Write-Host "Verifying required tools..." -ForegroundColor Cyan
pip install --upgrade pip build twine pytest pytest-cov

# 2. Verify all changes are committed before proceeding
if (git status --porcelain) {
    Write-Host "ERROR: There are uncommitted changes in the repository." -ForegroundColor Red
    Write-Host "Please commit all changes before running this script." -ForegroundColor Red
    exit 1
}

# 3. Create version update branch
$BranchName = "release/v$NewVersion"
git checkout -b $BranchName
Write-Host "Created branch: $BranchName" -ForegroundColor Green

# 4. Update version numbers in all necessary files
Write-Host "Updating version to $NewVersion in all files..." -ForegroundColor Cyan

# 4.1 Update version.py
$VersionFile = Join-Path $PackageRoot "me2ai_mcp\version.py"
if (Test-Path $VersionFile) {
    (Get-Content $VersionFile) -replace '__version__ = "[0-9]+\.[0-9]+\.[0-9]+"', "__version__ = `"$NewVersion`"" | Set-Content $VersionFile
    Write-Host "✓ Updated $VersionFile" -ForegroundColor Green
}

# 4.2 Update setup.py
$SetupFile = Join-Path $PackageRoot "setup.py"
if (Test-Path $SetupFile) {
    $SetupContent = Get-Content $SetupFile -Raw
    if ($SetupContent -match "VERSION = `"[0-9]+\.[0-9]+\.[0-9]+`"") {
        $SetupContent = $SetupContent -replace 'VERSION = "[0-9]+\.[0-9]+\.[0-9]+"', "VERSION = `"$NewVersion`""
    }
    elseif ($SetupContent -match "__version__ = `"[0-9]+\.[0-9]+\.[0-9]+`"") {
        $SetupContent = $SetupContent -replace '__version__ = "[0-9]+\.[0-9]+\.[0-9]+"', "__version__ = `"$NewVersion`""
    }
    $SetupContent | Set-Content $SetupFile
    Write-Host "✓ Updated $SetupFile" -ForegroundColor Green
}

# 4.3 Update pyproject.toml
$PyprojectFile = Join-Path $PackageRoot "pyproject.toml"
if (Test-Path $PyprojectFile) {
    (Get-Content $PyprojectFile) -replace 'version = "[0-9]+\.[0-9]+\.[0-9]+"', "version = `"$NewVersion`"" | Set-Content $PyprojectFile
    Write-Host "✓ Updated $PyprojectFile" -ForegroundColor Green
}

# 4.4 Verify license format in pyproject.toml
$PyprojectContent = Get-Content $PyprojectFile -Raw
if ($PyprojectContent -match 'license = "[^{]') {
    Write-Host "WARNING: License in pyproject.toml is not in object format!" -ForegroundColor Yellow
    Write-Host "Converting to proper format..." -ForegroundColor Yellow
    $PyprojectContent = $PyprojectContent -replace 'license = "([^"]+)"', 'license = { text = "$1 License" }'
    $PyprojectContent | Set-Content $PyprojectFile
    Write-Host "✓ Fixed license format in $PyprojectFile" -ForegroundColor Green
}

# 5. Update CHANGELOG.md
$ChangelogFile = Join-Path $PackageRoot "CHANGELOG.md"
if (Test-Path $ChangelogFile) {
    $ChangelogContent = Get-Content $ChangelogFile -Raw
    $CurrentDate = Get-Date -Format "yyyy-MM-dd"
    
    if ($ChangelogContent -match "## \[Unreleased\]") {
        $UpdatedChangelog = $ChangelogContent -replace "## \[Unreleased\]", "## [Unreleased]`n`n## [$NewVersion] - $CurrentDate"
        $UpdatedChangelog | Set-Content $ChangelogFile
        Write-Host "✓ Updated $ChangelogFile with new version" -ForegroundColor Green
        Write-Host "Remember to fill in the details for version $NewVersion in CHANGELOG.md" -ForegroundColor Yellow
    }
    else {
        Write-Host "WARNING: Could not find '## [Unreleased]' section in CHANGELOG.md" -ForegroundColor Yellow
        Write-Host "Please update CHANGELOG.md manually" -ForegroundColor Yellow
    }
}

# 6. Clean build artifacts
Write-Host "Cleaning previous build artifacts..." -ForegroundColor Cyan
Get-ChildItem -Path $PackageRoot -Include "__pycache__", "*.egg-info", "dist", "build" -Recurse -Force | Remove-Item -Force -Recurse
Write-Host "✓ Build artifacts cleaned" -ForegroundColor Green

# 7. Run tests (unless skipped)
if (-not $SkipTests) {
    Write-Host "Running tests..." -ForegroundColor Cyan
    python -m pytest --cov=me2ai_mcp
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Tests failed! Please fix tests before proceeding." -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ All tests passed" -ForegroundColor Green
}

# 8. Build package
Write-Host "Building package..." -ForegroundColor Cyan
python -m build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Package build failed!" -ForegroundColor Red
    exit 1
}

# 9. Verify build artifacts contain correct version
$DistFiles = Get-ChildItem -Path (Join-Path $PackageRoot "dist")
$VersionPattern = $NewVersion -replace '\.','\.'
$CorrectVersionFiles = $DistFiles | Where-Object { $_.Name -match $VersionPattern }

if ($CorrectVersionFiles.Count -eq 0) {
    Write-Host "ERROR: Built packages do not contain the specified version ($NewVersion)!" -ForegroundColor Red
    Write-Host "Found files:" -ForegroundColor Red
    $DistFiles | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}
Write-Host "✓ Build completed with correct version" -ForegroundColor Green

# 10. Upload to PyPI (ask for confirmation)
Write-Host "Ready to upload version $NewVersion to PyPI" -ForegroundColor Cyan
$Confirmation = Read-Host "Do you want to proceed with upload? (y/n)"
if ($Confirmation -ne 'y') {
    Write-Host "Upload canceled by user" -ForegroundColor Yellow
    exit 0
}

# 11. Get PyPI token from .env file
if (Test-Path $EnvFilePath) {
    $PyPIToken = (Get-Content $EnvFilePath | Where-Object { $_ -match "PYPI_API_TOKEN" } | ForEach-Object { $_.Split('=')[1].Trim() })
    if ([string]::IsNullOrEmpty($PyPIToken)) {
        Write-Host "ERROR: Could not find PYPI_API_TOKEN in $EnvFilePath" -ForegroundColor Red
        exit 1
    }
    $env:PYPI_API_TOKEN = $PyPIToken
} else {
    Write-Host "ERROR: .env file not found at $EnvFilePath" -ForegroundColor Red
    exit 1
}

# 12. Upload to PyPI
Write-Host "Uploading to PyPI..." -ForegroundColor Cyan
python -m twine upload dist/* --username __token__ --password $env:PYPI_API_TOKEN

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Upload to PyPI failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Successfully uploaded to PyPI" -ForegroundColor Green

# 13. Verify package availability (wait for PyPI indexing)
Write-Host "Waiting for PyPI to index the package..." -ForegroundColor Cyan
Start-Sleep -Seconds 60

# 14. Try installing the package
Write-Host "Verifying package installation..." -ForegroundColor Cyan
pip install --no-cache-dir "$PackageName==$NewVersion"

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Package verification failed. The package may not be available yet." -ForegroundColor Yellow
} else {
    Write-Host "✓ Package successfully installed from PyPI" -ForegroundColor Green
}

# 15. Create a git tag for the release
git add $VersionFile $SetupFile $PyprojectFile $ChangelogFile
git commit -m "Release version $NewVersion"
git tag -a "v$NewVersion" -m "Version $NewVersion"

Write-Host "✓ Created git commit and tag for version $NewVersion" -ForegroundColor Green
Write-Host "To push changes to remote repository, run:" -ForegroundColor Cyan
Write-Host "  git push origin $BranchName" -ForegroundColor Cyan
Write-Host "  git push origin v$NewVersion" -ForegroundColor Cyan

Write-Host "`nRelease process complete!" -ForegroundColor Green