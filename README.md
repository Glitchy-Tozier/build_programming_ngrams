# Programming Language N-Gram Corpora

This tool builds high-quality character-level n-gram frequency files (unigrams, bigrams, and trigrams) by analyzing the top 1,000 GitHub repositories for each of the 20 most popular programming languages.

The resulting corpora are intended for **keyboard layout optimization** projects (e.g. evaluating or training new layouts for programmers).

### What it does
1. Fetches the top 1,000 repositories per language via the GitHub Search API.
2. Clones only the default branch (shallow clone) and extracts files of the relevant language.
3. Counts unigrams, bigrams and trigrams **per repository**, normalizes them to proportions (so every repo contributes equally).
4. Averages the results per language and saves them as `1-grams.txt`, `2-grams.txt` and `3-grams.txt`.
5. Combines all languages into a final weighted corpus (`total_programming_ngrams`) using each language’s GitHub popularity as weight.

### Usage

```bash
python3.12 -m venv venv  # If necessary
source venv/bin/activate # If necessary
pip install tqdm requests # (if you don’t have them)

# Basic usage (recommended: provide a GitHub token)
python3 build_programming_ngrams.py --token YOUR_GITHUB_TOKEN

# Custom options
python3 build_programming_ngrams.py \
  --token YOUR_GITHUB_TOKEN \
  --languages "Python,JavaScript,Rust" \
  --num-repos 500 \
  --parallel 8 \
  --output-dir my_corpora
```

**Available options**
- `--languages` Comma-separated list (default: 20 most popular languages)
- `--num-repos`  Number of repos per language (max 1000)
- `--token`    GitHub personal access token (highly recommended; use classic, repo scope not needed — just public read)
- `--parallel`  Number of concurrent clones (default: 6)
- `--output-dir` Base output directory (default: `ngram_corpora`)

### Output structure
```
ngram_corpora/
├── Python_ngrams/
│   ├── 1-grams.txt
│   ├── 2-grams.txt
│   └── 3-grams.txt
├── JavaScript_ngrams/
│   └── ...
├── ...
└── total_programming_ngrams/
    ├── 1-grams.txt
    ├── 2-grams.txt
    └── 3-grams.txt
```

Each file contains lines in the format:
```
frequency ngram
```

Example:
```
0.0790694626    
0.0162750116 \n
0.0046266081 ion
```

The corpora are ready to be used directly in keyboard layout evaluation tools.
