# Setup guide: VS Code, Python environment, GitHub

Follow these in order. Commands are for **Windows (PowerShell)** and **macOS/Linux**; use the one that matches your machine.

## 1. Install the tools (one time)

1. **Python 3.10, 3.11 or 3.12** from https://www.python.org/downloads/. On Windows, tick **"Add python.exe to PATH"** in the installer.
   Check: `python --version`
2. **Git** from https://git-scm.com/downloads. Check: `git --version`
3. **VS Code** from https://code.visualstudio.com/
4. In VS Code, open the Extensions panel (Ctrl+Shift+X) and install **Python** (Microsoft). VS Code will also suggest the
   project's recommended extensions when you open the folder.
5. Tell Git who you are (one time):
   ```
   git config --global user.name "Your Name"
   git config --global user.email "you@university.edu"
   ```

## 2. Create the GitHub repository

1. Sign in at https://github.com → top-right **+** → **New repository**.
2. Name: `fdia-gnn`. Visibility: **Private** (research code before publication).
3. **Do not** tick "Add a README", ".gitignore" or "license" — the project already has them, and an empty repo avoids a merge conflict on the first push.
4. Click **Create repository** and keep the page open; it shows the repo URL, e.g. `https://github.com/<you>/fdia-gnn.git`.

## 3. Put the project on your computer

1. Unzip `fdia-gnn.zip` somewhere permanent, e.g. `Documents/research/fdia-gnn`.
2. VS Code → **File → Open Folder…** → select the `fdia-gnn` folder.
3. Open a terminal inside VS Code: **Terminal → New Terminal**. It opens in the project folder.

## 4. Create the Python environment

```
python -m venv .venv
```
Activate it:
- Windows: `.venv\Scripts\activate`  (if blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry)
- macOS/Linux: `source .venv/bin/activate`

Your prompt now starts with `(.venv)`. Then:
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```
torch is the big download (~200 MB–2 GB). If you have an NVIDIA GPU and want to use it, install torch from
https://pytorch.org/get-started/locally/ instead; it is optional — 118 buses train fine on CPU.

In VS Code: **Ctrl+Shift+P → "Python: Select Interpreter"** → pick the one in `.venv`.

## 5. Check it works

```
python -m pytest
```
Expect `13 passed`. The beaker icon in the left bar (Testing) shows the same tests; you can run them there too.

## 6. Connect to GitHub and push

In the VS Code terminal, inside the project folder:
```
git init
git add .
git commit -m "Initial test bed: data pipeline, GNN baselines, tests, first results"
git branch -M main
git remote add origin https://github.com/<you>/fdia-gnn.git
git push -u origin main
```
The first push opens a browser window to sign in to GitHub. Refresh the GitHub page: your files appear, including
`data/fdia_dataset_ieee118.npz` (~10 MB).

## 7. Add your professor as a collaborator

GitHub repo page → **Settings → Collaborators → Add people** → enter his GitHub username or email → he gets an invite.
With write access he can comment and edit; if you only want him to see it, that's still the same "collaborator"
invite on a personal private repo.

## 8. Daily workflow

```
git pull                          # get any changes he made
# ...work, run experiments...
git add .
git commit -m "Short description of what changed"
git push
```
VS Code's **Source Control** panel (Ctrl+Shift+G) does the same with buttons.

Good habits:
- Commit `results/*.md` and `results/*.json` after each experiment so he can see numbers without running anything.
- The main dataset (~10 MB) is committed. If a future dataset grows past ~50 MB (e.g. 40,000+ samples), stop committing it
  and share it through Box / Teams instead; GitHub rejects files over 100 MB.
- Put a line in the README **Status** table each week.

## 9. Sharing the dataset

It's in the repo: once you push, your professor gets it with `git pull` (or the GitHub "Download" button on the file).
Anyone can also regenerate it identically:
```
python scripts/generate_data.py --samples 3000 --workers 2 --seed 42
```
(Same seed **and** same `--workers` value gives the same data.)
