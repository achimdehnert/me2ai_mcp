# PyPI Token Generation Instructions

Follow these steps to create a proper PyPI token:

## For the Official PyPI Repository

1. Go to https://pypi.org/manage/account/
2. Sign in to your PyPI account
3. Navigate to Account Settings → API tokens
4. Click "Add API token"
5. Set the token scope:
   - Select "Entire account (all projects)" if you want to publish multiple packages
   - Or select "Project: me2ai_mcp" for this specific package only
6. Create the token and copy it immediately (you won't see it again)
7. Update your `.env` file with:
   ```
   PYPI_API_TOKEN=pypi-xxxx...  # Your actual token
   ```

## For TestPyPI (Testing)

1. Go to https://test.pypi.org/manage/account/
2. Sign in to your TestPyPI account (may require separate registration)
3. Navigate to Account Settings → API tokens
4. Click "Add API token"
5. Set token scope as described above
6. Create the token and copy it immediately
7. Update your `.env` file with:
   ```
   TEST_PYPI_API_TOKEN=pypi-xxxx...  # Your actual TestPyPI token
   ```

## Using the Token

```powershell
# For official PyPI
$env:TWINE_USERNAME="__token__"
$env:TWINE_PASSWORD=$env:PYPI_API_TOKEN
python -m twine upload dist/*

# For TestPyPI
$env:TWINE_USERNAME="__token__"
$env:TWINE_PASSWORD=$env:TEST_PYPI_API_TOKEN
python -m twine upload --repository testpypi dist/*
```
