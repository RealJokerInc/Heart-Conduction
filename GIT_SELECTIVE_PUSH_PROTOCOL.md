# Git Selective Push Protocol

## Purpose
This protocol handles pushing specific file types from directories that are otherwise ignored by `.gitignore`, particularly for managing course files where only code/notebooks should be tracked while excluding datasets, artifacts, and other large files.

## Common Scenario
You need to push only `.py` and `.ipynb` files from a directory (e.g., "BME 4790") that contains many other files you don't want to track (datasets, images, virtual environments, etc.).

---

## Step-by-Step Protocol

### 1. Check for Nested Git Repositories
**Problem:** Subdirectories with their own `.git` folders are treated as submodules and won't be added normally.

**Check:**
```bash
find "TARGET_DIRECTORY" -name ".git" -type d
```

**Fix if found:**
```bash
# Remove nested .git directories (CAUTION: This removes git history)
rm -rf "TARGET_DIRECTORY/SUBDIRECTORY/.git"
```

**⚠️ Warning:** Only do this if the nested repository isn't needed. Consider backing up the repository first if it contains important history.

---

### 2. Update .gitignore to Allow Specific File Types

**Current pattern (blocks everything):**
```gitignore
BME 4790/
```

**Updated pattern (allows only .py and .ipynb):**
```gitignore
# Exclude large project folders (but allow .py and .ipynb files)
BME 4790/**
!BME 4790/**/*.py
!BME 4790/**/*.ipynb
!BME 4790/**/
!BME 4790/
```

**Explanation:**
- `BME 4790/**` - Ignore everything in BME 4790
- `!BME 4790/**/*.py` - Except Python files
- `!BME 4790/**/*.ipynb` - Except Jupyter notebooks
- `!BME 4790/**/` - Except directories (needed for subdirectories)
- `!BME 4790/` - Except the root folder itself

---

### 3. Find All Target Files

**List all .py files (excluding virtual environments):**
```bash
find "TARGET_DIRECTORY" -name "*.py" -not -path "*/.*" -not -path "*/.venv/*" -not -path "*/venv/*"
```

**List all .ipynb files:**
```bash
find "TARGET_DIRECTORY" -name "*.ipynb" -not -path "*/.*"
```

---

### 4. Stage the Files

**Method 1: Using bash glob (recommended):**
```bash
bash -c 'git add "TARGET_DIRECTORY"/**/*.py "TARGET_DIRECTORY"/**/*.ipynb'
```

**Method 2: Using find + xargs:**
```bash
find "TARGET_DIRECTORY" \( -name "*.py" -o -name "*.ipynb" \) -not -path "*/.*" -not -path "*/.venv/*" -print0 | xargs -0 git add
```

**Method 3: Individual subdirectories:**
```bash
git add "TARGET_DIRECTORY/SUBDIR1"/*.py "TARGET_DIRECTORY/SUBDIR1"/*.ipynb
git add "TARGET_DIRECTORY/SUBDIR2"/*.py "TARGET_DIRECTORY/SUBDIR2"/*.ipynb
```

---

### 5. Verify Staged Files

**Check what will be committed:**
```bash
git status --short | grep "TARGET_DIRECTORY"
```

**See detailed diff:**
```bash
git diff --cached --stat
```

**Count files by type:**
```bash
git diff --cached --name-only | grep -E "\.(py|ipynb)$" | wc -l
```

---

### 6. Commit and Push

**Commit with descriptive message:**
```bash
git commit -m "Add [description] files (.py and .ipynb only)

Added Python scripts and Jupyter notebooks from:
- [List major sections/folders]

Updated .gitignore to allow .py and .ipynb files while excluding datasets and artifacts.
"
```

**Push to GitHub:**
```bash
git push
```

---

## Troubleshooting

### Issue: Files still won't stage
**Cause:** Files are already tracked in git cache as ignored

**Solution:**
```bash
git rm -r --cached "TARGET_DIRECTORY"
git add "TARGET_DIRECTORY"/**/*.py "TARGET_DIRECTORY"/**/*.ipynb
```

### Issue: "No matches found" error
**Cause:** Shell doesn't expand globs correctly

**Solution:** Use `bash -c` wrapper:
```bash
bash -c 'git add "TARGET_DIRECTORY"/**/*.py'
```

### Issue: Too many files or files with special characters
**Solution:** Use find with null-terminated output:
```bash
find "TARGET_DIRECTORY" -name "*.py" -print0 | xargs -0 git add
```

### Issue: Virtual environment files are being added
**Solution:** Add to .gitignore:
```gitignore
**/.venv/
**/venv/
**/__pycache__/
```

---

## Quick Reference Commands

**Full workflow for new directory:**
```bash
# 1. Check for nested repos
find "TARGET_DIR" -name ".git" -type d

# 2. Remove if needed (backup first!)
rm -rf "TARGET_DIR/SUBDIR/.git"

# 3. Update .gitignore (manually edit)

# 4. Stage files
bash -c 'git add "TARGET_DIR"/**/*.py "TARGET_DIR"/**/*.ipynb'

# 5. Verify
git status --short | grep "TARGET_DIR"

# 6. Commit
git commit -m "Add TARGET_DIR files (.py and .ipynb only)"

# 7. Push
git push
```

---

## Prevention for Future Projects

**When creating new project directories:**

1. **Don't initialize git inside subdirectories** if they're part of a larger repo
2. **Create .gitignore upfront** with selective patterns
3. **Document file organization** so you know what should/shouldn't be tracked
4. **Use virtual environments** in standard locations (.venv, venv) that are globally ignored

**Template .gitignore for course directories:**
```gitignore
# Course folder - only track code
COURSE_NAME/**
!COURSE_NAME/**/*.py
!COURSE_NAME/**/*.ipynb
!COURSE_NAME/**/*.md
!COURSE_NAME/**/

# Always exclude
**/.venv/
**/venv/
**/__pycache__/
**/*.pyc
*.zip
*.csv
*.png
*.jpg
*.jpeg
```

---

## Notes

- This protocol preserves large files (datasets, images, etc.) locally while only tracking code
- Selective .gitignore patterns work recursively through all subdirectories
- Always verify with `git status` before committing to avoid accidentally adding large files
- Consider using `.gitattributes` for line ending normalization in .ipynb files
