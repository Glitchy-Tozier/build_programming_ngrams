#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm

# ==================== CONFIG ====================
DEFAULT_LANGUAGES = [
    "JavaScript", "Python", "Java", "TypeScript", "C++", "C#", "Go", "Rust",
    "PHP", "Ruby", "C", "Swift", "Kotlin", "Dart", "Objective-C", "Scala",
    "Lua", "Haskell", "Perl", "Elixir"
]

LANGUAGE_EXTENSIONS: Dict[str, List[str]] = {
    "JavaScript": [".js", ".jsx", ".mjs", ".cjs"],
    "Python": [".py", ".pyi", ".pyx", ".pxd"],
    "Java": [".java"],
    "TypeScript": [".ts", ".tsx"],
    "C++": [".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".hh", ".h"],
    "C#": [".cs"],
    "Go": [".go"],
    "Rust": [".rs"],
    "PHP": [".php"],
    "Ruby": [".rb"],
    "C": [".c", ".h"],
    "Swift": [".swift"],
    "Kotlin": [".kt", ".kts"],
    "Dart": [".dart"],
    "Objective-C": [".m", ".mm", ".h"],
    "Scala": [".scala"],
    "Lua": [".lua"],
    "Haskell": [".hs", ".lhs"],
    "Perl": [".pl", ".pm"],
    "Elixir": [".ex", ".exs"],
}

IGNORED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__",
    "build", "dist", "target", "bin", "obj", "out", ".idea", ".vscode"
}

# ===============================================

def github_search_repos(lang: str, num_repos: int, token: str | None) -> Tuple[List[str], int]:
    """Return up to num_repos repo full_names + total_count."""
    repos: List[str] = []
    total_count = 0
    per_page = 100
    pages = (num_repos + per_page - 1) // per_page

    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    for page in range(1, pages + 1):
        params = {
            "q": f"language:{lang.lower()}",
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
            "page": page,
        }
        resp = requests.get(
            "https://api.github.com/search/repositories",
            headers=headers,
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if page == 1:
            total_count = data.get("total_count", 0)
        items = data.get("items", [])
        if not items:
            break
        repos.extend(item["full_name"] for item in items)
        if len(repos) >= num_repos:
            break
        # respect search rate limit
        if not token:
            time.sleep(2)  # unauthenticated is stricter

    return repos[:num_repos], total_count


def process_single_repo(full_name: str, lang: str, extensions: List[str]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Clone, extract language files, compute normalised n-gram proportions."""
    uni_repo = defaultdict(float)
    bi_repo = defaultdict(float)
    tri_repo = defaultdict(float)

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(tmp) / full_name.split("/")[-1]
        try:
            # shallow clone of default branch
            subprocess.run(
                ["git", "clone", "--depth", "1", "--single-branch", f"https://github.com/{full_name}.git", str(repo_dir)],
                check=True,
                capture_output=True,
                timeout=90,
            )

            # walk and process only relevant files
            for root, dirs, files in os.walk(repo_dir):
                # prune ignored directories
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

                for file in files:
                    if any(file.endswith(ext) for ext in extensions):
                        filepath = Path(root) / file
                        try:
                            text = filepath.read_text(encoding="utf-8", errors="replace")
                            chars = text.replace("\r", "")
                            if not chars:
                                continue

                            # unigrams
                            for c in chars:
                                uni_repo[c] += 1
                            # bigrams
                            for i in range(len(chars) - 1):
                                bi_repo[chars[i : i + 2]] += 1
                            # trigrams
                            for i in range(len(chars) - 2):
                                tri_repo[chars[i : i + 3]] += 1
                        except Exception:
                            continue  # skip unreadable file

            # normalise to proportions (each repo contributes equally)
            def normalise(d: Dict[str, float]) -> Dict[str, float]:
                total = sum(d.values())
                return {k: v / total for k, v in d.items()} if total > 0 else {}

            return normalise(uni_repo), normalise(bi_repo), normalise(tri_repo)

        except Exception as e:
            print(f"    ⚠️  Failed {full_name}: {e}")
            return {}, {}, {}


def save_ngrams(ngrams: Dict[str, float], path: Path, n: int):
    if not ngrams:
        return
    sorted_ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)
    with open(path, "w", encoding="utf-8") as f:
        for gram, freq in sorted_ngrams:
            # escape control characters but keep unicode and space as-is
            display = gram.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            f.write(f"{freq:.10f} {display}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate character n-gram corpora for programming languages from GitHub.")
    parser.add_argument("--languages", type=str, default=None,
                        help="Comma-separated list of languages (default: top 20)")
    parser.add_argument("--num-repos", type=int, default=1000,
                        help="Number of top repos per language (max 1000 due to GitHub search limit)")
    parser.add_argument("--token", type=str, default=None,
                        help="GitHub personal access token (recommended)")
    parser.add_argument("--output-dir", type=Path, default=Path("ngram_corpora"),
                        help="Base output directory")
    parser.add_argument("--parallel", type=int, default=6,
                        help="Number of concurrent repo clones (default 6)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallelism")
    args = parser.parse_args()

    token = args.token or os.getenv("GITHUB_TOKEN")
    languages = [l.strip() for l in args.languages.split(",")] if args.languages else DEFAULT_LANGUAGES
    out_base = args.output_dir
    out_base.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting programming-language n-gram corpus builder")
    print(f"   Languages : {', '.join(languages)}")
    print(f"   Repos per language : {args.num_repos}")
    print(f"   Output directory : {out_base.resolve()}\n")

    lang_corpus: Dict[str, Tuple[Dict[str, float], Dict[str, float], Dict[str, float], int]] = {}

    for idx, lang in enumerate(languages, 1):
        print(f"[{idx}/{len(languages)}] 🔍 Processing {lang} ...")
        repos, total_count = github_search_repos(lang, args.num_repos, token)
        print(f"    Found {total_count:,} total repos → using top {len(repos)} (popularity weight = {total_count:,})")

        extensions = LANGUAGE_EXTENSIONS.get(lang, [".txt"])  # fallback
        uni_total = defaultdict(float)
        bi_total = defaultdict(float)
        tri_total = defaultdict(float)
        processed = 0

        def worker(repo: str):
            return repo, process_single_repo(repo, lang, extensions)

        futures = []
        with ThreadPoolExecutor(max_workers=1 if args.no_parallel else args.parallel) as executor:
            for repo in repos:
                futures.append(executor.submit(worker, repo))

            for future in tqdm(as_completed(futures), total=len(repos), desc="    Cloning & counting", unit="repo"):
                repo_name, (u, b, t) = future.result()
                for g, v in u.items():
                    uni_total[g] += v
                for g, v in b.items():
                    bi_total[g] += v
                for g, v in t.items():
                    tri_total[g] += v
                processed += 1

        # average the proportions
        n = processed if processed > 0 else 1
        for d in (uni_total, bi_total, tri_total):
            for k in list(d.keys()):
                d[k] /= n

        lang_dir = out_base / f"{lang}_ngrams"
        lang_dir.mkdir(parents=True, exist_ok=True)
        save_ngrams(uni_total, lang_dir / "1-grams.txt", 1)
        save_ngrams(bi_total, lang_dir / "2-grams.txt", 2)
        save_ngrams(tri_total, lang_dir / "3-grams.txt", 3)

        lang_corpus[lang] = (uni_total, bi_total, tri_total, total_count)
        print(f"    ✅ Saved {lang} n-grams ({processed} repos)\n")

    # ====================== COMBINE ======================
    print("🔗 Combining all languages into total_programming_ngrams ...")
    total_uni = defaultdict(float)
    total_bi = defaultdict(float)
    total_tri = defaultdict(float)
    total_weight = sum(w for _, _, _, w in lang_corpus.values())

    for lang, (u, b, t, weight) in lang_corpus.items():
        w = weight / total_weight if total_weight > 0 else 0
        for g, v in u.items():
            total_uni[g] += v * w
        for g, v in b.items():
            total_bi[g] += v * w
        for g, v in t.items():
            total_tri[g] += v * w

    total_dir = out_base / "total_programming_ngrams"
    total_dir.mkdir(parents=True, exist_ok=True)
    save_ngrams(total_uni, total_dir / "1-grams.txt", 1)
    save_ngrams(total_bi, total_dir / "2-grams.txt", 2)
    save_ngrams(total_tri, total_dir / "3-grams.txt", 3)

    print(f"🎉 Done! Everything saved to {out_base.resolve()}")
    print("   You can now run your normalisation script on any *_ngrams folder if you want percentages.")


if __name__ == "__main__":
    main()