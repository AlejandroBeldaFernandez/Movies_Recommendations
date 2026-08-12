# Movie Recommendation System

A movie recommendation system built with three independent, comparable approaches: content
based filtering, item based collaborative filtering, and matrix factorization (SVD). All three
are evaluated on the same held out data and compared directly, with their failure modes and
cold start behavior documented rather than hidden.

- **Problem:** Recommend movies from three genuinely different signals (metadata, audience
  co-rating behavior, and learned latent factors), covering both an existing user or movie and
  the cold start case where neither exists yet
- **Result:** Collaborative filtering scores Precision@5 42.41% and Recall@5 15.16% against
  content based filtering's 5.93%/1.87% and SVD's 6.05%/1.57%, a gap of roughly 7 to 8 times
  that the notebook explains rather than just reports
- **Value:** A complete, audited recommendation pipeline, not a single trained model: real bugs
  found and fixed with evidence at every stage, cold start handled explicitly for two of three
  models, and an honest account of what offline metrics can and cannot prove on a dataset this
  size

> [Ver este proyecto en español](README_ES.md)

---

## Live demo

**[Try it live on Hugging Face Spaces](https://huggingface.co/spaces/Alessandrou24/movie__Recommender)**

Pick one or more movies and compare what the three models recommend side by side, or describe a
movie that isn't in the catalog for a content based match.

> **If the Space is slow to load or shows an error, be patient before assuming it's broken.** It
> runs on Hugging Face's free ZeroGPU tier, which puts the Space to sleep after a period with no
> visitors to save shared resources. Opening the link wakes it up automatically, but the first
> request after a period of inactivity can take a minute or two while it restarts — that's
> expected, not a bug.
>
> If it still doesn't come back:
> 1. Open the Space page and check the status badge at the top (`Running` / `Building` /
>    `Sleeping` / `Runtime error`).
> 2. If it's stuck on `Sleeping`, refresh the page or wait ~30-60 seconds after your first
>    request — that alone resolves it most of the time.
> 3. If it shows `Runtime error` or `Build error`, open the **Logs** tab on the Space page to see
>    why, or go to **Settings → Restart this Space** to force a clean restart.
> 4. Restarting does not retrain or recompute anything: the three models run from precomputed
>    artifacts, so a restart just reloads them and is safe to do at any time.

---

## Table of contents

- [Live demo](#live-demo)
1. [Problem definition](#problem-definition)
2. [Dataset](#dataset)
3. [Data cleaning](#data-cleaning)
4. [Exploratory data analysis](#exploratory-data-analysis)
5. [Content based filtering](#content-based-filtering)
6. [Collaborative filtering](#collaborative-filtering)
7. [Matrix factorization (SVD)](#matrix-factorization-svd)
8. [Evaluation](#evaluation)
9. [How each model decides](#how-each-model-decides)
10. [Problems found](#problems-found)
11. [Business conclusions](#business-conclusions)
12. [Who this is for](#who-this-is-for)
13. [Project value](#project-value)
14. [Possible improvements](#possible-improvements)
15. [Requirements](#requirements)

---

## Problem definition

Recommend movies to a user given one of two kinds of input: a movie they already like ("more
like this"), or their own rating history ("for you"). Three models are built to answer this,
each relying on a different signal:

| Model | Signal used |
|---|---|
| Content based | Genre, director, cast, keywords, production company, franchise |
| Collaborative (item based) | Which users rated which movies, and how |
| SVD | Latent factors learned from the same ratings |

A recommender that only works for well established users and titles has limited real world
value, so the project also covers the cold start case explicitly: a user with no rating
history, and a movie with no interactions.

---

## Dataset

[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (Kaggle,
compiled from TMDB and MovieLens), downloaded via `kagglehub`.

| File | Content |
|---|---|
| `movies_metadata.csv` | ~45,000 movies: title, genres, budget, revenue, runtime, language, release date, production companies/countries, votes, overview |
| `credits.csv` | Cast and crew per movie (JSON encoded), used for director and lead actors |
| `keywords.csv` | Plot keywords per movie (JSON encoded) |
| `links_small.csv` | Mapping between MovieLens `movieId`, IMDb `imdbId` and TMDB `tmdbId` |
| `ratings_small.csv` | User ratings (`userId`, `movieId`, `rating`, `timestamp`) |

The full `ratings.csv` (about 26 million ratings) does not fit in memory locally, so this
project uses the small versions throughout. That decision fixes the project's scope: after
cleaning and deduplication, the catalog is **7,818 movies**, and `ratings_small.csv` covers
**671 users** with roughly **100,000 ratings** (16 to 1,391 per user).

---

## Data cleaning

Two independent joins (metadata with credits/keywords, and metadata with links/ratings) merge
into one rating level table, cleaned before either model is built:

- JSON encoded columns (`genres`, `cast`, `crew`, `keywords`, `production_companies`,
  `belongs_to_collection`) parsed with `ast.literal_eval` into usable Python structures.
- Type fixes: `budget` and `revenue` coerced to numeric, `release_date` to datetime, `imdb_id`
  stripped of its `tt` prefix.
- Duplicates and near empty rows dropped; a movie missing every content relevant column at once
  is excluded rather than kept with placeholder text.
- `production_countries` and `spoken_languages` dropped early: both are heavily concentrated
  (about 70% USA, about 95% English) and carry little discriminative signal, and dropping them
  before the blank value check means a blank in either can no longer cost an otherwise complete
  movie its place in the dataset.

Several real bugs were found and fixed during this stage: a `.median` call missing its
parentheses silently corrupting numeric columns, `budget` never converted to numeric,
`belongs_to_collection` reverting to null after JSON parsing, and `crew` being joined character
by character instead of by list element.

---

## Exploratory data analysis

Eleven questions, each with a stated hypothesis, a statistical test where one applies, and an
effect size, not just a p-value. Full detail and code are in the notebook; headline findings:

- **Genres.** Drama (4,038) and Comedy (2,889) dominate; Foreign (34) and TV Movie (24) are
  nearly absent, too few to support meaningful genre based comparisons.
- **Budget.** Heavily right skewed (median \$20M, mean \$33.65M); 43.4% of the catalog has no
  reported budget at all.
- **Budget vs. revenue.** Positively correlated, Pearson r = 0.71 (r² = 0.51) excluding zeros.
- **vote_average vs. vote_count.** Only a weak relationship (r² ≈ 0.04 to 0.05); a movie's
  popularity says very little about how well it is rated.
- **Language.** English dominates at 88.1%, driven by Hollywood's production volume and a
  Western skewed sampling bias in the dataset, not by speaker count (Mandarin has more).
- **Budget tier vs. rating.** Low budget movies rate higher in 64.1% of pairwise comparisons
  (Mann-Whitney, small to moderate effect).
- **Language × decade.** Statistically significant (p = 3.5e-32) but practically negligible
  (Cramér's V = 0.074), a reminder that significance and effect size answer different questions.
- **vote_count by decade.** The strongest finding in the EDA (Kruskal-Wallis, η² = 0.218): newer
  decades have far more votes, an artifact of platform growth, not of movie quality.

---

## Content based filtering

Each movie's `genres`, `director`, `cast`, `keywords`, `production_companies` and
`belongs_to_collection` are combined into one normalized text string (`text`), lowercased and
stripped of internal spaces so a full name counts as one token instead of splitting into common
words. Franchise membership turned out to be one of the strongest signals available, stronger
than genre for that specific case, so standalone movies get their own title as a stand in.

`CountVectorizer` and `TfidfVectorizer` are both built and compared. On movies already in the
catalog they agree closely on the strong signal (Toy Story returns Toy Story 2/3 under both) and
diverge on the tail: querying `Sabrina (1995)`, `CountVectorizer` returns other classic era
romantic comedies, while `TfidfVectorizer` returns Sydney Pollack's own filmography, the
director of that exact movie, because rare tokens are weighted far above common genre words.

**Cold start** for a movie that does not exist in the catalog at all works: `vectorizer.transform()`
reuses the fitted vocabulary without ever refitting anything, and unknown tokens are simply
ignored. Tested with *Dune: Part Two*, released after this dataset was built. A `threshold`
parameter (default `0.1` for the cold start function, `0.0` for the in catalog one) guards
against the case where a hand written text shares no vocabulary with the corpus at all: without
it, a similarity of exactly `0.0` against every movie is still a valid sort order, so the
function would silently return whatever movies happened to sit last after the tie break, dressed
up as a real recommendation.

---

## Collaborative filtering

A user-item matrix (`userId` × `movieId`, values are ratings, `0` for unrated) is built from
`ratings_small.csv`, and movie-to-movie similarity is computed as the cosine similarity between
columns of that matrix, entirely independent of any metadata. Zero fill is safe here because
ratings never go to `0` on this dataset's 0.5-to-5.0 scale, and because a zero contributes
nothing to either the dot product or the vector norm, so the comparison is effectively computed
only over users who rated both movies.

Movies with fewer than 5 ratings are filtered out before computing similarity: with a single
rating, a movie's similarity to whatever that one user also rated comes out at or near 1.0,
regardless of whether the movies are actually alike, a false perfect match rather than a weak
signal. That cutoff drops the catalog from 7,818 movies to **3,357 (43%)**, the single largest
structural limitation of this model.

**Cold start for a new user** is solved: a short list of titles the user says they like is
treated as equally weighted pseudo ratings, reusing the same weighted sum of similarity rows
used for an existing user's real ratings. **Cold start for a new movie is not solved**, and
cannot be from inside this model: a title only earns a place in the similarity matrix once real
people have rated it enough times. That gap is documented as a stated limitation, not patched
with a fallback to content based filtering; the two models are kept independent by design so the
trade-off between them stays visible.

---

## Matrix factorization (SVD)

Built with `surprise`'s `SVD` as a third, independent model: every user and every movie get a
vector of latent factors, and a predicted rating is `global_mean + user_bias + movie_bias +
dot(user_vector, movie_vector)`. Unlike a generic dense decomposition, `surprise` trains only on
ratings that actually exist, never on the zeros the collaborative model relies on as a filler.

RMSE on held out ratings is **0.8735**, in the range typically reported for SVD on MovieLens
sized data (the Netflix Prize's baseline scored 0.95, the winning entry 0.857 after years of
tuning). That number alone says nothing about personalization: testing on two very different
users (an eclectic viewer and a mainstream one) showed both pulled toward the same small pool of
universally acclaimed classics before `random_state` was fixed, and the overlap narrowed but did
not disappear once it was. With as few as 16 ratings for some users, the personalized term in
the formula has little to learn from, so a movie's own general popularity dominates the
prediction more than in either model above.

Cold start for a new user is a harder gap here than for collaborative filtering: `svd.predict()`
for an unseen user id falls back to the global mean rating for every movie, a prediction that
looks valid but carries no personalization at all, and closing it needs either a full retrain or
a fold-in step this project does not implement.

---

## Evaluation

All three models are compared with Precision@5 and Recall@5 on the exact same held out data per
user: 80% of each user's ratings are used as input, the rest are the test set, and a movie
counts as relevant if it was rated `3` or higher, the midpoint of the 0.5-to-5.0 scale.

| Model | Precision@5 | Recall@5 |
|---|---|---|
| Collaborative (item based) | 42.41% | 15.16% |
| SVD | 6.05% | 1.57% |
| Content based | 5.93% | 1.87% |

Collaborative filtering wins by roughly 7 to 8 times on both metrics, and this is expected
rather than surprising: the relevance signal here is a held out rating, exactly the task
collaborative filtering is built to predict and exactly the task content based filtering was
never given access to. The gap says which model predicts *held out ratings* better, not which
one a real user would act on, a different claim that would need real users shown real
recommendations to answer. It also does not mean content based recommendations are wrong: a
movie a user never rated is not evidence they would dislike it, only that they likely never saw
it, so precision@k *systematically underestimates* any model good at surfacing something a user
has not encountered yet, which is exactly the behavior a useful recommender needs.

---

## How each model decides

| | Content based | Collaborative (item based) | SVD |
|---|---|---|---|
| Input it needs | A movie's own metadata | Ratings from many users | Ratings from many users |
| Core computation | Cosine similarity between TF-IDF/count vectors | Cosine similarity between rating columns | `dot(user_vector, movie_vector)` plus bias terms |
| What "similar" means | Same kind of object (genre, cast, franchise) | Watched by the same people | Close in a learned, unlabeled factor space |
| New movie | Works immediately from a text description | Impossible until it accumulates ratings | Impossible until it accumulates ratings |
| New user | Works from one liked title | Works from a few liked titles | Falls back to the global average, no personalization |
| Typical failure | Recommends more of the same, never surprises | False perfect matches on very thin data | Converges toward generically popular titles regardless of who is asking |

The three are kept independent by design, none falls back to another when it cannot answer, so
the trade-off between them stays visible instead of being hidden behind a single blended list.

---

## Problems found

A representative sample of real, evidence backed bugs found and fixed while building this
project, each confirmed by reproducing it before fixing it:

- **Duplicate titles broke lookups.** Remakes and reboots sharing a title (Sabrina 1954/1995,
  Ghostbusters 1984/2016) made a title-to-id lookup return more than one row, raising
  `KeyError`/`ValueError`. Fixed by appending the release year to every duplicated title.
- **A silent failure worse than a crash.** A cold start query with no vocabulary overlap at all
  produced a cosine similarity of exactly `0.0` against every movie, which is still a valid sort
  order, so the function returned whatever movies happened to sit last after the tie break,
  looking like a real recommendation. Fixed with a `threshold` parameter that returns
  `'No movies found'` instead.
- **`SVD()` without `random_state` was not reproducible.** Two very different users shared
  several of their top recommendations across separate model fits before this was found, and the
  overlap changed on every retrain, independent of any actual signal in the data.
- **A coordinated train/test split mattered more than it looked.** SVD was first evaluated
  against a split drawn independently of the one it was trained on, meaning it could get credit
  for a rating it had already seen during training. Fixed by deriving both from the same split.
- **`.loc` vs. plain `[]` on a DataFrame.** Selecting rows with a list of row labels through
  plain `[]` is read by pandas as a request for columns, not rows, a mistake that surfaced more
  than once across both models and always failed loudly enough to catch, but easily could not
  have.

---

## Business conclusions

- **Two of three models solve their hardest cold start case; none solves all of it.** Content
  based filtering has no cold start problem at all, since it never needs interaction data.
  Collaborative filtering and SVD both solve the new user case, but neither can recommend a movie
  that has not yet accumulated real ratings, a structural gap tied to how each model is built,
  not something a parameter change fixes.
- **Higher accuracy on this offline metric is not the same as a better product.** Collaborative
  filtering's 7 to 8 times advantage on Precision@5/Recall@5 reflects that the metric measures
  the exact task it was built for. A product decision between "more like this" and "for you"
  should be driven by what the user is trying to do, not by which model wins an offline number
  computed on a different question.
- **43% catalog coverage is the real cost of the collaborative model's strength.** The same
  5-rating cutoff that keeps its similarities trustworthy also means 57% of the catalog cannot be
  recommended, or queried, through it. Any product built on this model needs a plan for that
  majority of the catalog, whether that is content based filtering, a popularity fallback, or
  both.
- **The evaluation numbers here answer a narrower question than "which model is best."** They
  say which model best predicts a rating that already happened, not which recommendation a real
  user would act on. Closing that gap for real needs the same thing any production recommender
  needs before its quality can be claimed with confidence: real users, shown real
  recommendations, with their next action observed afterward.

---

## Who this is for

**Individuals.** A "for you" list built from your own rating history (collaborative or SVD), or
a "more like this" list from a single movie you already liked (content based), including titles
outside the dataset entirely if you can describe them, useful for discovery beyond what a
platform already knows you watched.

**Companies.** A streaming or rental platform gets three deployable strategies rather than one:
collaborative filtering for the 43% of the catalog with enough engagement data, content based
filtering for everything else including new releases the moment their metadata exists, and a
documented, quantified account of the trade-off between them to decide where each is worth the
engineering cost. The same architecture (independent models, explicit cold start handling,
matched offline evaluation) generalizes past movies to any catalog with both rich item metadata
and user interaction data: books, music, e-commerce products, articles.

---

## Project value

This covers the full cycle of a recommendation project end to end, not a single model trained
and left in a notebook: data cleaning and EDA grounded in hypothesis testing rather than visual
impression, three distinct recommendation approaches with their failure modes documented rather
than hidden, formal evaluation with standard retrieval metrics together with an explicit account
of what those metrics can and cannot settle on a dataset this size, and a from scratch account of
every real bug found along the way, not just the code that worked. That last part is the
differentiator: most of what makes this project trustworthy is visible in the notebook precisely
because the mistakes were kept in it, not edited out.

---

## Possible improvements

**Modeling**

- **Autoencoders.** A deep autoencoder trained to reconstruct a rating vector can capture
  non-linear interaction patterns cosine similarity cannot, and tends to outperform classical
  collaborative filtering once there is enough data to train one.
- **Text embeddings for content.** `text` here is a bag of tokens from structured metadata, with
  `overview` deliberately excluded. Sentence embeddings from a language model over the actual
  plot summary would capture thematic similarity no keyword list does.
- **LLMs as a component.** An LLM could re-rank or explain a shortlist already produced by these
  models, or act as a judge in place of the offline metrics this project could not use fairly
  between models.
- **A properly blended hybrid.** This project keeps the three models independent by design so
  the trade-off stays visible; a production system might instead blend them, trading that
  visibility for a single, better performing list.

**Data**

- **The full dataset.** With 25 million ratings instead of 100,000, the 5-rating minimum
  enforced for collaborative filtering could likely be lowered substantially, shrinking or
  closing the 43% coverage gap.
- **Temporal signal.** `timestamp` was dropped early and never used; recency is a real signal a
  static model like this one ignores entirely.
- **Online evaluation.** The one thing no offline metric here can substitute for: real users,
  shown real recommendations, with their next action observed.

---

## Requirements

```
pandas==3.0.5
numpy==2.5.1
matplotlib==3.11.1
seaborn==0.13.2
scipy==1.18.0
scikit-learn==1.9.0
scikit-posthocs==0.14.0
scikit-surprise==1.1.5
kagglehub==1.0.2
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

Python 3.12 was used to build this project.

---

*Dataset source: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset*
